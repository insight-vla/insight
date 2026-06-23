"""xArm + RealSense lifecycle and low-level control.

Replaces the module-level ``arm`` / ``camera_pipelines`` / ``FPS`` globals in
``inference_primitives`` with an instance-scoped class. Holds the connection
state for one run; ``close()`` releases the pipelines on exit.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from insight.rotation import quat_to_axis_angle, rpy_to_quat_deg
from xarm.wrapper import XArmAPI


SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "317222072257"

_CONTROL_HZ = 40.0

# Canonical home pose (mm + deg). Matches the spacemouse teleop's go_home()
# target so the arm starts every trial from the same configuration the user
# tunes via the spacemouse, regardless of where the previous trial left it.
_HOME_POSE = (465.4, 0.0, 388.7, -178.1, 0.0, -179.9)

# Hard ceiling on TCP translation speed for skill-gap motion. Acts as a
# code-side backstop independent of ``--max-tcp-speed-mm-s``: the arg is
# still honored when smaller, and used when 0 (otherwise-disabled) gets
# substituted with this value. Prevents accidental high-speed motion if
# the user forgets the flag or passes a too-large override.
_MAX_TCP_SPEED_CEILING_MM_S = 120.0


def _R_from_xarm_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    """Rotation matrix R that maps EE-local vectors to world frame.

    xArm reports pose as ``(x, y, z, roll, pitch, yaw)`` in mm/deg with the
    ZYX intrinsic Euler convention (rotate by ``roll`` about local X, then
    ``pitch`` about new Y, then ``yaw`` about new Z), which composes to:

        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Used by ``XArmHardware.tool_tip_pose`` to transform the EE-local tool
    offset into a world-frame position for FK-based tip-floor enforcement.
    """
    roll, pitch, yaw = rpy_rad
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)


class XArmHardware:
    """Owns the xArm + camera pipelines + control rate.

    All hardware-touching methods (``capture_frames``, ``interpolate_action``,
    ``get_pose``, etc.) go through an instance so the executor can be tested
    or swapped without touching module globals.
    """

    def __init__(self, arm_ip: str, fps: float,
                 workspace_bounds: tuple[float, ...] = (),
                 max_tcp_speed_mm_s: float = 0.0,
                 tool_offset_local_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 tool_tip_floor_z_mm: float = 0.0,
                 use_gripper: bool = True,
                 use_angle_axis_control: bool = False):
        self.arm_ip = arm_ip
        self.fps = fps
        self.dt = 1.0 / fps
        self._control_dt = 1.0 / _CONTROL_HZ
        self.arm: XArmAPI | None = None
        self._pipelines: dict[str, rs.pipeline] = {}
        # Bounds expected as (x_min, x_max, y_min, y_max, z_min, z_max) or empty
        # to disable. Sanity-check shape up front so a malformed tuple fails at
        # connect time, not silently mid-run.
        if workspace_bounds and len(workspace_bounds) != 6:
            raise ValueError(
                f"workspace_bounds must be 6 values or empty, got {len(workspace_bounds)}"
            )
        self._bounds = tuple(workspace_bounds) if workspace_bounds else ()
        self._clamp_state: tuple = ()
        self._speed_clamp_active = False
        self._speed_clamp_worst = 1.0
        # Effective TCP speed cap: clamp the arg against the hard ceiling.
        # Treat 0 (the arg's "disabled" sentinel) as "use the ceiling" so the
        # backstop fires even when --max-tcp-speed-mm-s is omitted entirely.
        if max_tcp_speed_mm_s <= 0:
            self._max_tcp_speed = _MAX_TCP_SPEED_CEILING_MM_S
        else:
            self._max_tcp_speed = min(float(max_tcp_speed_mm_s), _MAX_TCP_SPEED_CEILING_MM_S)
            if max_tcp_speed_mm_s > _MAX_TCP_SPEED_CEILING_MM_S:
                logging.warning(
                    "max_tcp_speed_mm_s=%.1f capped to hard ceiling %.1f mm/s",
                    max_tcp_speed_mm_s, _MAX_TCP_SPEED_CEILING_MM_S,
                )
        # FK-based tool-tip floor enforcement (currently DORMANT — disabled by
        # default args). Kept around because the orientation-induced error of
        # the simpler TCP-Z floor is small for current tools (~5mm at 26°
        # drift on a 50mm tool), so the simpler approach suffices. Re-enable
        # by passing non-zero --tool-offset-mm and --tool-tip-floor-z-mm if
        # tool length grows or orientation variance becomes significant.
        self._tool_offset_local = np.array(tool_offset_local_mm, dtype=np.float32)
        self._has_tool_offset = bool(np.any(self._tool_offset_local))
        self._tool_tip_floor_z = float(tool_tip_floor_z_mm)
        # Whether the end-effector is a controllable gripper. Gates gripper
        # init in connect(), command dispatch in the runner's policy loop,
        # and gripper-action recording.
        self.use_gripper = bool(use_gripper)
        # Send servo commands via set_servo_cartesian_aa (axis-angle) instead
        # of set_servo_cartesian (rpy). Eliminates parameterization branch
        # ambiguity at gimbal lock — equivalent rpy values collapse to the
        # same rotation vector at the SDK boundary.
        self._use_angle_axis = bool(use_angle_axis_control)
        # Canonical home pose for every trial. Matches spacemouse teleop's
        # go_home() target.
        self.home_pose = np.array(_HOME_POSE, dtype=np.float32)

    # ────────────────── Lifecycle ──────────────────

    def connect(self) -> None:
        """Connect to the arm and start the cameras. Idempotent.

        On first connect, also drives the arm to the canonical home pose so
        every batch starts from the same configuration the spacemouse teleop
        uses for ``go_home()`` — independent of where the previous user
        session left the arm."""
        first_connect = self.arm is None
        if first_connect:
            self.arm = XArmAPI(self.arm_ip)
            self.arm.connect()
            if self.arm.get_state() != 0:
                self.arm.clean_error()
                time.sleep(0.5)
            self.arm.motion_enable(enable=True)
            # Sensitivity 5 (max) trips error=31 on normal grasping forces during
            # pickplace primitives. 3 keeps reasonable collision protection
            # without being so tight that lifting a grasped object is rejected.
            self.arm.set_collision_sensitivity(3)
            if self._bounds:
                self._enable_reduced_mode()
            self.arm.set_mode(1)
            self.arm.set_state(0)
            if self.use_gripper:
                # Gripper bring-up. Without explicit enable, the SDK silently
                # ignores set_gripper_position calls. Skipped on non-gripper
                # end-effectors (e.g. scoop) where these calls would error.
                self.arm.clean_gripper_error()
                self.arm.set_gripper_mode(0)    # 0 = position/location mode
                self.arm.set_gripper_enable(True)
            if self._bounds:
                self._warn_if_pose_out_of_bounds()
        if not self._pipelines:
            self._start_cameras()
        if first_connect:
            logging.info("[HOME] driving arm to canonical home: %s",
                         self.home_pose.tolist())
            self.return_to_home(self.home_pose)

    def _enable_reduced_mode(self) -> None:
        """Set the xArm's hardware-side workspace bounds.

        SDK call order is: ``set_reduced_mode(True)`` → ``set_reduced_tcp_boundary``
        with ``[x_max, x_min, y_max, y_min, z_max, z_min]`` (note max-then-min
        per axis, opposite of how we store it) → optional ``set_reduced_max_tcp_speed``.
        Once enabled, the arm itself refuses any servo command outside the box.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
        self.arm.set_reduced_mode(True)
        self.arm.set_reduced_tcp_boundary([x_max, x_min, y_max, y_min, z_max, z_min])
        if self._max_tcp_speed > 0:
            self.arm.set_reduced_max_tcp_speed(self._max_tcp_speed)
        logging.info("Reduced mode enabled: x=[%.0f,%.0f] y=[%.0f,%.0f] z=[%.0f,%.0f] mm",
                     x_min, x_max, y_min, y_max, z_min, z_max)
        if self._max_tcp_speed > 0:
            logging.info("Reduced-mode max TCP speed: %.0f mm/s", self._max_tcp_speed)

    def _warn_if_pose_out_of_bounds(self) -> None:
        """Sanity-check current pose vs configured bounds — early failure beats
        a confusing servo refusal mid-run."""
        x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
        pose = self.get_pose()
        x, y, z = pose[0], pose[1], pose[2]
        out_of_bounds = (
            x < x_min or x > x_max
            or y < y_min or y > y_max
            or z < z_min or z > z_max
        )
        if out_of_bounds:
            logging.warning(
                "Initial pose (%.1f, %.1f, %.1f) is OUTSIDE workspace bounds. "
                "Move the arm into bounds before running motion, or reduced mode "
                "may refuse the first command.", x, y, z,
            )

    def _start_cameras(self) -> None:
        for serial in [SERIAL_EXTERNAL, SERIAL_WRIST]:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            # Wrist captures at the camera's native 16:9 aspect (the D435i
            # sensor is native 16:9; the previous 320x240 4:3 mode cropped
            # the horizontal FOV). 640x360 is a modest size that preserves
            # native FOV. capture_frames() crops+resizes back to 320x240
            # 4:3 for the policy / recorder so their inputs are unchanged;
            # the raw 640x360 frame is cached on self for VLM use.
            if serial == SERIAL_WRIST:
                cfg.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
            else:
                cfg.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
            try:
                pipeline.start(cfg)
                self._pipelines[serial] = pipeline
                logging.info("Started camera: %s", serial)
            except Exception as e:
                logging.error("Failed to start camera %s: %s", serial, e)
                sys.exit(1)

    def close(self) -> None:
        """Release cameras. Arm cleanup is the caller's responsibility."""
        for pipeline in self._pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        self._pipelines.clear()

    # ────────────────── Sensing ──────────────────

    def capture_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest (exterior, wrist) RGB frames. Blocks briefly.

        The wrist arrives natively at 640x360 (16:9) — the camera's native
        aspect, preserving full horizontal FOV. We crop+resize to the
        legacy 320x240 (4:3) shape for the policy / recorder so their
        inputs are unchanged. The full 640x360 frame is cached on
        ``self._last_wrist_full`` for VLM-only consumers that want the
        wider view (e.g., pre-analysis skill-gap motion).

        RealSense cameras occasionally disconnect under load (USB bandwidth,
        cable strain during arm motion). Rather than crashing the whole trial,
        we catch the disconnect and attempt to restart the affected pipeline
        up to a few times before giving up. If reconnection fails, propagate
        the original error.
        """
        f_ext = self._wait_for_frames_with_retry(SERIAL_EXTERNAL)
        f_wrist = self._wait_for_frames_with_retry(SERIAL_WRIST)
        exterior = np.asanyarray(f_ext.get_color_frame().get_data()).copy()
        wrist_full = np.asanyarray(f_wrist.get_color_frame().get_data()).copy()
        # Cache for VLM consumers.
        self._last_wrist_full = wrist_full
        # Policy-compat: center-crop 640x360 → 480x360 (4:3), resize → 320x240.
        h, w = wrist_full.shape[:2]
        target_w = int(round(h * 4.0 / 3.0))
        x0 = (w - target_w) // 2
        cropped = wrist_full[:, x0:x0 + target_w, :]
        wrist_policy = cv2.resize(cropped, (320, 240), interpolation=cv2.INTER_AREA)
        return exterior, wrist_policy

    @property
    def last_wrist_full(self) -> np.ndarray | None:
        """The last raw wrist frame at the camera's native 16:9 aspect.
        Populated by ``capture_frames``; ``None`` before the first capture."""
        return getattr(self, "_last_wrist_full", None)

    # ────────────── Continuous (uncut) capture ─────────────────────

    def start_continuous_capture(self, fps: float = 10.0) -> None:
        """Begin a background thread that captures (ext, wrist) at ~``fps``
        Hz into ``self._continuous_frames`` until ``stop_continuous_capture``.

        The point is to record wall-clock-faithful video that INCLUDES the
        time spent inside VLM calls (planner, pre-analysis, oracle) and
        VLA inference — periods where the main capture loop is blocked
        and the trial's stepwise frames have gaps. The result is saved
        as a separate ``all_primitives_continuous_*.mp4`` alongside the
        existing stepwise video.

        The background thread shares the RealSense pipelines with the
        main loop. RealSense frame buffers absorb the contention — if
        main is mid-call, the background just gets the next available
        frame. No new streams or pipelines are opened.
        """
        # Reset state. Caller is responsible for retrieving frames via
        # ``stop_continuous_capture`` BEFORE starting another trial.
        self._continuous_frames: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._continuous_stop = threading.Event()
        interval = 1.0 / max(fps, 1.0)

        def _loop() -> None:
            while not self._continuous_stop.is_set():
                t = time.perf_counter()
                try:
                    ext, wrist = self.capture_frames()
                    self._continuous_frames.append((t, ext, wrist))
                except Exception as e:
                    # A failed wait_for_frames (timeout, disconnect) is
                    # logged but doesn't kill the trial; capture resumes
                    # on the next iteration. ``capture_frames`` already
                    # retries internally on RuntimeError.
                    logging.debug("[CONTINUOUS] capture skipped: %s", e)
                # Sleep just enough to hit the target fps. Drift is fine —
                # we're not trying to be metronome-accurate.
                elapsed = time.perf_counter() - t
                self._continuous_stop.wait(max(0.0, interval - elapsed))

        self._continuous_thread = threading.Thread(target=_loop, daemon=True)
        self._continuous_thread.start()

    def stop_continuous_capture(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Stop the background thread and return the captured frames.

        Returns a list of ``(timestamp_s, ext_frame, wrist_frame)`` tuples
        ordered by capture time. Empty if ``start_continuous_capture`` was
        never called. Safe to call multiple times.
        """
        stop_event = getattr(self, "_continuous_stop", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(self, "_continuous_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)
        return getattr(self, "_continuous_frames", [])

    def _wait_for_frames_with_retry(self, serial: str, max_retries: int = 3):
        """``wait_for_frames`` with auto-restart on disconnect.

        Treats any RuntimeError from wait_for_frames as a recoverable
        camera-side issue (disconnect, frame timeout, stale pipeline state,
        etc.) and rebuilds the pipeline from scratch. Bails out only after
        max_retries failed restarts.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._pipelines[serial].wait_for_frames()
            except RuntimeError as e:
                last_err = e
                logging.warning("[CAMERA] %s read failed (%s); attempt %d/%d to rebuild pipeline...",
                                serial, str(e)[:100], attempt + 1, max_retries)

            # Try to stop the broken pipeline (best effort).
            try:
                self._pipelines[serial].stop()
            except Exception:
                pass
            time.sleep(1.0)  # let USB settle before re-enumerating

            # Build + start a fresh pipeline. Only swap into _pipelines on success.
            new_pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            # Match the initial _start_cameras config: wrist at 16:9 native
            # (640x360), external at legacy 4:3 (320x240).
            if serial == SERIAL_WRIST:
                cfg.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
            else:
                cfg.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
            try:
                new_pipeline.start(cfg)
                self._pipelines[serial] = new_pipeline
                logging.info("[CAMERA] %s reconnected.", serial)
                time.sleep(0.5)  # warmup before first wait
            except Exception as start_err:
                logging.error("[CAMERA] %s restart failed: %s — likely USB cable/power issue.", serial, start_err)
                last_err = start_err
                time.sleep(2.0)  # back off before next attempt
                # Leave a sentinel so next iteration immediately retries the start
                # rather than calling wait_for_frames on a stopped pipeline.
                self._pipelines[serial] = new_pipeline  # not started, but tracked

        # Exhausted retries — give the caller the last error.
        if last_err is not None:
            raise last_err

    def get_pose(self) -> np.ndarray:
        """Current arm pose as ``[x, y, z, roll, pitch, yaw]`` (mm + deg).

        Roll/yaw are wrapped to ``[0, 360)`` to match the training-data convention.
        """
        assert self.arm is not None
        pose = list(self.arm.get_position()[1])
        pose[3] %= 360
        pose[5] %= 360
        return np.array(pose, dtype=np.float32)

    def gripper_norm(self) -> float:
        """Current gripper position in the policy's normalized convention.

        Inverts the runner's ``gripper_target = 850 - 860 * action[6]`` so the
        recorded value lines up with what the policy outputs at the same
        physical state. Returns ``0.0`` when ``use_gripper=False`` so callers
        can pass the result unconditionally without branching.
        """
        if not self.use_gripper:
            return 0.0
        assert self.arm is not None
        raw = self.arm.get_gripper_position()[1]
        return float((850.0 - float(raw)) / 860.0)

    # FK helpers — currently dormant. Active only when ``tool_offset_local_mm``
    # is non-zero. See ``__init__`` comment for rationale.

    def tool_tip_pose(self, tcp_pose: np.ndarray | None = None) -> np.ndarray:
        """Tool tip world XYZ (mm), via FK from a TCP pose. Returns the TCP
        XYZ unchanged when no tool offset has been configured."""
        if tcp_pose is None:
            tcp_pose = self.get_pose()
        if not self._has_tool_offset:
            return tcp_pose[:3].copy()
        R = _R_from_xarm_rpy(np.deg2rad(tcp_pose[3:6]))
        return tcp_pose[:3] + R @ self._tool_offset_local

    def tool_tip_z(self, tcp_pose: np.ndarray | None = None) -> float:
        """Tool tip world-Z (mm). Convenience wrapper around ``tool_tip_pose``."""
        return float(self.tool_tip_pose(tcp_pose)[2])

    def build_observation(self, prompt: str, exterior: np.ndarray, wrist: np.ndarray) -> dict:
        """Assemble the observation dict the policy server expects."""
        pose = self.get_pose()
        angles_rad = (pose[3:6] * np.pi / 180).tolist()
        state = np.array(pose[:3].tolist() + angles_rad, dtype=np.float32)
        return {
            "observation/exterior_image_1_left": exterior,
            "observation/wrist_image_left": wrist,
            "observation/state": state,
            "prompt": prompt,
        }

    # ────────────────── Actuation ──────────────────

    # Canonical joint configuration at the home EE pose. Captured directly
    # from the arm at the user's "initial pose." Driving by joint angles
    # (instead of EE-target via set_position) makes the wrist branch
    # deterministic: same physical pose can be reached as joint-6 = +180.67°
    # or -179.33°, and set_position was nondeterministically picking either,
    # causing the wrist to sometimes rotate the long way around.
    HOME_JOINT_ANGLES_DEG = (
        -0.278801, -1.635451, -64.543237, -0.646526, 66.166484, 180.674620,
    )

    def return_to_home(self, home: np.ndarray, *,
                       speed_mm_s: float = 50.0,
                       joint_speed_deg_s: float = 12.0) -> None:
        """Drive the arm to the canonical home joint configuration.

        Switches into position-control mode (mode 0), calls ``set_servo_angle``
        with the canonical joint vector, then switches back into servo mode
        (mode 1) for the next primitive. Targeting joints (not EE) avoids the
        wrist-branch ambiguity that ``set_position`` was hitting at the
        ±180° yaw boundary. ``home`` (EE pose) is accepted but unused; we keep
        the param for API compatibility.

        We deliberately don't reuse the primitive servo loop here: tracking a
        time-varying policy target at 40Hz is a different problem from "go to
        a fixed known pose."
        """
        del home  # unused — we drive by joint angles instead
        assert self.arm is not None

        # Open the gripper fully before driving home so we don't carry residual
        # grip from a previous trial (e.g. the bottle from a partial pour).
        # Position 850 = fully open per the collect_demo convention. Skipped on
        # non-gripper end-effectors where set_gripper_position would error.
        if self.use_gripper:
            try:
                self.arm.set_gripper_position(850, wait=False)
            except Exception as e:
                logging.warning("[RETURN-HOME] gripper open failed: %s", e)

        # Switch into position-control mode for the planned move.
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.2)  # let the mode switch settle before issuing commands

        code = self.arm.set_servo_angle(
            angle=list(self.HOME_JOINT_ANGLES_DEG),
            speed=joint_speed_deg_s, wait=True,
        )
        if code != 0:
            logging.warning("[RETURN-HOME] set_servo_angle returned code=%d", code)
        else:
            logging.info("[RETURN-HOME] reached home via set_servo_angle.")

        # Restore servo mode for the next trial's primitives.
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.2)

    def begin_planned_motion(self) -> None:
        """Enter position-control mode (mode 0) for controller-planned trajectories.

        Same control path xArm Studio's web teleop uses for cartesian jog
        buttons: the controller plans a smooth linear-cartesian trajectory
        with accel/decel profiles, rather than tracking discrete servo
        targets at 40Hz. Used by the rotation skill-gap branch so unscrew /
        pour-tilt motions look like web-teleop arcs instead of the visibly
        rough servo-mode discrete-target chase.

        Caller MUST pair this with ``end_planned_motion()`` to restore
        servo mode for the next primitive — typically via try/finally.
        """
        assert self.arm is not None
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.2)

    def end_planned_motion(self) -> None:
        """Restore servo mode (mode 1) after a planned-motion section.

        Waits for the controller to finish the in-progress trajectory
        before switching modes. We exit the polling loop as soon as the
        quat-error gap drops below threshold, but the planned trajectory
        may still be settling — calling ``set_mode(1)`` while ``get_state()``
        still reports 1 (in motion) returns code=10 and silently leaves
        the arm in mode 0. Subsequent servo commands then fail with
        'mode may be incorrect' warnings and don't actuate.
        """
        assert self.arm is not None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            code, state = self.arm.get_state()
            # state == 1 means in motion. Anything else (2 sleeping,
            # 3 suspended, 4 stopping, or 0 ready) means safe to switch.
            if code != 0 or state != 1:
                break
            time.sleep(0.05)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.2)

    def interpolate_action(self, state: np.ndarray, goal: np.ndarray,
                           z_floor: float, divisor: float = 6.0) -> None:
        """Drive the arm from ``state`` toward ``goal`` for one outer-loop tick.

        Uses ``divisor`` as a damping factor (1 = no damping, higher = smoother).
        Z is clamped to ``z_floor`` and (if configured) X/Y/Z to ``workspace_bounds``
        before each servo command. If ``max_tcp_speed_mm_s`` was set at
        construction, the per-tick translation is additionally scaled down to
        respect that velocity cap (defends against the xArm's reduced-mode TCP
        speed setting not enforcing as expected on large step deltas).
        """
        assert self.arm is not None
        inner_iters = int(self.dt * _CONTROL_HZ)
        delta = (goal - state) / (self.dt * _CONTROL_HZ * divisor)

        # Software speed clamp. Per-outer-tick motion is ``inner_iters * delta``
        # because the inner loop ramps from ``state + delta`` to ``state +
        # inner_iters * delta`` (see incremental-target loop below). Cap that
        # against ``max_tcp_speed * dt``.
        if self._max_tcp_speed > 0:
            per_tick_dist_mm = float(np.linalg.norm(delta[:3])) * inner_iters
            max_dist_mm = self._max_tcp_speed * self.dt
            if per_tick_dist_mm > max_dist_mm:
                scale = max_dist_mm / per_tick_dist_mm
                if not self._speed_clamp_active:
                    logging.info("[SPEED-CLAMP] engaged: cap %.1fmm/tick", max_dist_mm)
                    self._speed_clamp_active = True
                    self._speed_clamp_worst = scale
                else:
                    self._speed_clamp_worst = min(self._speed_clamp_worst, scale)
                delta[:3] = delta[:3] * scale
            elif self._speed_clamp_active:
                logging.info("[SPEED-CLAMP] cleared (worst %.2f×)", self._speed_clamp_worst)
                self._speed_clamp_active = False
                self._speed_clamp_worst = 1.0

        # Incremental targets at 40Hz — instead of sending ``state + delta`` four
        # times (which jerks 25% toward goal then holds for 3 servo ticks), ramp
        # smoothly: 25% → 50% → 75% → 100% across the outer tick. Lets the xArm
        # interpolate continuously rather than step-and-hold.
        for i in range(inner_iters):
            t0 = time.perf_counter()
            command = state + delta * (i + 1)
            # Unwrap roll/yaw to be within 180° of the arm's CURRENT state
            # (not the principal range). The xArm controller interprets RPY
            # values literally for joint planning -- if state.roll = +182°
            # and we command -178°, it plans a 360° joint swing for the
            # same physical pose. Keeping the command in the same branch as
            # state avoids that. Backward-compatible for translation
            # primitives where command[3]/[5] equal state[3]/[5] anyway.
            for axis in (3, 5):
                diff = command[axis] - state[axis]
                command[axis] -= 360.0 * round(diff / 360.0)
            if command[2] < z_floor:
                logging.info("[Z-FLOOR-CLAMP] %.2f -> %.2f", command[2], z_floor)
                command[2] = z_floor
            if self._bounds:
                x_min, x_max, y_min, y_max, z_min, z_max = self._bounds
                edges, key = [], []
                if command[0] < x_min:
                    edges.append(f"x={command[0]:.1f}<x_min={x_min:.1f}")
                    key.append("x<min"); command[0] = x_min
                elif command[0] > x_max:
                    edges.append(f"x={command[0]:.1f}>x_max={x_max:.1f}")
                    key.append("x>max"); command[0] = x_max
                if command[1] < y_min:
                    edges.append(f"y={command[1]:.1f}<y_min={y_min:.1f}")
                    key.append("y<min"); command[1] = y_min
                elif command[1] > y_max:
                    edges.append(f"y={command[1]:.1f}>y_max={y_max:.1f}")
                    key.append("y>max"); command[1] = y_max
                if command[2] < z_min:
                    edges.append(f"z={command[2]:.1f}<z_min={z_min:.1f}")
                    key.append("z<min"); command[2] = z_min
                elif command[2] > z_max:
                    edges.append(f"z={command[2]:.1f}>z_max={z_max:.1f}")
                    key.append("z>max"); command[2] = z_max
                new_state = tuple(key)
                if new_state != self._clamp_state:
                    if edges:
                        logging.info("[BOUND-CLAMP] %s", ", ".join(edges))
                    else:
                        logging.info("[BOUND-CLAMP] cleared")
                    self._clamp_state = new_state
            # Tool-tip floor clamp (FK-based) — DORMANT by default. Activates
            # when both ``tool_offset_local_mm`` and ``tool_tip_floor_z_mm``
            # are non-zero. Raises commanded TCP-Z so the FK-derived tool tip
            # stays at or above the configured world-Z floor regardless of EE
            # orientation. Currently unused since the simpler TCP-Z workspace
            # bound is sufficient for current tools.
            if self._tool_tip_floor_z > 0 and self._has_tool_offset:
                R = _R_from_xarm_rpy(np.deg2rad(command[3:6]))
                z_offset_world = float(R[2, :] @ self._tool_offset_local)
                tip_z = command[2] + z_offset_world
                if tip_z < self._tool_tip_floor_z:
                    new_z = self._tool_tip_floor_z - z_offset_world
                    logging.info(
                        "[TIP-FLOOR-CLAMP] tip_z=%.2f < floor=%.2f, raising tcp z=%.2f -> %.2f",
                        tip_z, self._tool_tip_floor_z, command[2], new_z,
                    )
                    command[2] = new_z
            if self._use_angle_axis:
                # rpy → quat → axis-angle (rotation vector, radians). Equivalent
                # rpy parameterizations of the same physical orientation collapse
                # to the same rotation vector here, giving the SDK an
                # unambiguous orientation target → consistent IK seed across
                # ticks → no joint flipping at gimbal lock.
                q = rpy_to_quat_deg(command[3:6])
                axis, angle = quat_to_axis_angle(q)
                rotvec = axis * angle
                pose_aa = [
                    float(command[0]), float(command[1]), float(command[2]),
                    float(rotvec[0]), float(rotvec[1]), float(rotvec[2]),
                ]
                self.arm.set_servo_cartesian_aa(
                    pose_aa, speed=100, mvacc=1000,
                    is_radian=True, relative=False,
                )
            else:
                self.arm.set_servo_cartesian(command, speed=100, mvacc=1000)
            time_left = self._control_dt - (time.perf_counter() - t0)
            time.sleep(max(time_left, 0))
