"""Post-plan success oracle + atomic commit/discard of deferred recordings.

Mixin used by ``XArmFlywheelExecutor``. Provides:

- ``_finalize_skill_gap_recording`` — drive to oracle viewing pose,
  capture after-frame, run the VLM success oracle, prompt for optional
  manual override, then atomically commit or discard all pending
  episodes accumulated during the trial.
- ``_prompt_verdict_override`` — small CLI helper for the manual-verdict
  flag; accepts/forces success/failure.

These methods are split from the main executor because they're
plumbing-heavy (pose driving, image capture, VLM call, gripper state
read, file IO for oracle frames, recorder commit/discard) and run only
once per trial — keeping them in a dedicated module makes the main
executor file easier to follow.
"""

from __future__ import annotations

import logging
import time

from insight.reasoning import check_task_completion


class _OracleMixin:
    """Post-plan oracle + verdict-override methods.

    Host class must provide: ``self.args``, ``self.hardware``,
    ``self.recorder``, ``self.run_dir``, ``self._pending_episodes``,
    ``self._home_pose``, ``self._before_frame``, ``self._timed_vlm``,
    ``self._timed_input``, ``self._last_trial_reason``,
    ``self._last_episodes_committed``, ``self._last_frames_committed``,
    ``self._last_oracle_overridden``.
    """

    # ────────────────── Post-plan success oracle ──────────────────

    def _finalize_skill_gap_recording(self) -> tuple[bool, str]:
        """Return to home, run the visual-diff success oracle, then commit or
        discard ALL deferred episodes (skill-gap + known primitives) together.

        Always runs return-to-home + oracle so eval trials (no skill-gap, no
        recording — the trained-policy test path) still get a proper
        success/failure verdict. When ``_pending_episodes`` is non-empty,
        the oracle additionally gates atomic commit/discard so the dataset
        only contains mutually consistent episode pairs — e.g., we never
        save a move-to-rocks episode whose sweep counterpart failed.

        Returns ``(success, reason)``."""
        if self.args.dry_run or self._home_pose is None or self._before_frame is None:
            self._pending_episodes = []
            return False, "dry run / no hardware"

        try:
            # Priority: --no-return-home-for-oracle is the master "capture in
            # place" switch and overrides oracle_pose. This matters for tasks
            # like pour where the gripper finishes in a specific pose (tilted
            # bottle over bowl) that the oracle needs to see — driving to any
            # viewing pose would disturb the held bottle.
            if not self.args.return_home_for_oracle:
                logging.info("\n[POST-PLAN] skipping pose change (--no-return-home-for-oracle); capturing after-frame in place.")
            elif self.args.oracle_pose:
                if len(self.args.oracle_pose) != 6:
                    raise ValueError(
                        f"oracle_pose must be 6 joint angles (deg), got {len(self.args.oracle_pose)}"
                    )
                logging.info("\n[POST-PLAN] driving to oracle viewing pose (joints) %s...",
                             list(self.args.oracle_pose))
                # Match return_to_home: switch to mode 0 (position-control)
                # before set_servo_angle, restore mode 1 (servo) after. Use
                # the same speed to avoid jerky motion. Joint-space target
                # avoids the ±180° wrist-branch ambiguity that set_position
                # hits when commanding Cartesian poses near yaw=±180°.
                #
                # Timed into _last_return_home_time_s — same bucket as the
                # canonical return-to-home, since this is also a deterministic
                # post-trial drive to a viewing pose (not productive
                # manipulation). Keeps the paper-reported Exec column honest
                # regardless of whether the eval used the home pose or a
                # custom oracle_pose.
                _t_drive = time.perf_counter()
                self.hardware.arm.set_mode(0)
                self.hardware.arm.set_state(0)
                time.sleep(0.2)
                code = self.hardware.arm.set_servo_angle(
                    angle=list(self.args.oracle_pose),
                    speed=self.args.home_joint_speed_deg_s, wait=True,
                )
                if code != 0:
                    logging.warning("[POST-PLAN] oracle-pose set_servo_angle returned code=%d", code)
                self.hardware.arm.set_mode(1)
                self.hardware.arm.set_state(0)
                time.sleep(0.2)
                self._last_return_home_time_s += time.perf_counter() - _t_drive
            else:
                logging.info("\n[POST-PLAN] returning to home for success oracle...")
                self._timed_return_home(
                    self._home_pose,
                    joint_speed_deg_s=self.args.home_joint_speed_deg_s,
                )
            after_ext, _ = self.hardware.capture_frames()

            # Save the exact frames the oracle is comparing so the user can
            # eyeball them when the verdict is wrong (most common: cap landed
            # outside the home-view, oracle says "still capped" because the
            # cap isn't visible in the after-frame).
            if self.run_dir is not None:
                import imageio.v2 as imageio
                oracle_dir = self.run_dir / "oracle_frames"
                oracle_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(str(oracle_dir / "before.jpg"), self._before_frame)
                imageio.imwrite(str(oracle_dir / "after.jpg"), after_ext)
                logging.info("[POST-PLAN] saved oracle before/after to %s", oracle_dir)

            # Pass ground-truth gripper state to the oracle alongside the
            # images. The VLM is unreliable at judging gripper open/closed
            # from a single after-frame (especially with thin grippers or
            # cluttered backgrounds), so feeding the sensor reading directly
            # reduces oracle false-negatives like "robot didn't release"
            # when the gripper actually IS released.
            gripper_extra = ""
            if self.hardware.use_gripper:
                try:
                    code, grip_raw = self.hardware.arm.get_gripper_position()
                    if code == 0 and grip_raw is not None:
                        # xArm convention: 0 = fully closed, 850 = fully open.
                        # Threshold at 500: anything above is unambiguously
                        # open enough to have released the object.
                        state_str = "OPEN" if grip_raw > 500 else "CLOSED"
                        gripper_extra = (
                            f"Final gripper state: {state_str} "
                            f"(raw position {int(grip_raw)}/850; "
                            f"0=fully closed, 850=fully open)."
                        )
                except Exception as e:
                    logging.warning("[POST-PLAN] gripper state read failed: %s", e)

            total_frames = sum(len(frames) for _, frames in self._pending_episodes)

            if not self.args.use_oracle:
                # Skip the VLM oracle entirely. The human is the sole judge.
                # We still drove to the viewing pose and captured + saved the
                # after-frame above, so oracle_frames/ has debug images even
                # without the VLM verdict.
                logging.info("[POST-PLAN] oracle disabled (--no-use-oracle); asking user for verdict.")
                self._last_trial_reason = "human-only: pending prompt"
                committed, reason_suffix = self._prompt_human_only_verdict(total_frames)
                self._last_oracle_overridden = False  # no oracle to override
                reason = "human-only: " + reason_suffix
                self._last_trial_reason = reason
            else:
                verdict = self._timed_oracle(
                    check_task_completion,
                    self.args.goal, self._before_frame, after_ext,
                    extra_context=gripper_extra,
                )
                logging.info("[POST-PLAN] task=%r completed=%s — %s",
                             self.args.goal, verdict["completed"], verdict["reasoning"])

                # Set the reason eagerly (before the verdict prompt) so a Ctrl+C
                # during the prompt still leaves a recoverable reason for the
                # batch logger to read. Full reasoning, no truncation —
                # JSONL row stays self-contained for paper analysis.
                self._last_trial_reason = "oracle: " + str(verdict.get("reasoning", ""))

                committed = bool(verdict["completed"])
                override_used = False
                if self.args.manual_verdict:
                    new_committed = self._prompt_verdict_override(committed, total_frames)
                    override_used = (new_committed != committed)
                    committed = new_committed

                self._last_oracle_overridden = override_used
                prefix = "user-override: " if override_used else "oracle: "
                reason = prefix + str(verdict.get("reasoning", ""))
                self._last_trial_reason = reason

            if not committed:
                if self._pending_episodes:
                    logging.info("[POST-PLAN] dropping all %d pending episodes (%d frames total)",
                                 len(self._pending_episodes), total_frames)
                self._pending_episodes = []
                self._last_episodes_committed = 0
                self._last_frames_committed = 0
                return False, reason
            if self._pending_episodes and self.recorder is not None:
                for primitive, frames in self._pending_episodes:
                    self.recorder.commit_buffered(frames)
                logging.info("[POST-PLAN] committed %d episodes (%d frames total)",
                             len(self._pending_episodes), total_frames)
            self._last_episodes_committed = len(self._pending_episodes)
            self._last_frames_committed = total_frames
            self._pending_episodes = []
            return True, reason
        except KeyboardInterrupt:
            logging.info("[POST-PLAN] interrupted; pending episodes NOT committed.")
            self._pending_episodes = []
            raise
        except Exception as e:
            logging.error("[POST-PLAN] oracle/return-to-home failed: %s", e)
            self._pending_episodes = []
            return False, f"oracle/return-to-home error: {e}"

    def _prompt_verdict_override(self, oracle_verdict: bool, n_frames: int) -> bool:
        """Ask the user to accept or override the oracle's success verdict.

        ENTER (or any non-y/n input) accepts the oracle. 'y' forces commit,
        'n' forces discard. Used to catch oracle false positives/negatives
        before the recording is locked in (e.g., when lift_upward
        accidentally swept and fooled the visual diff)."""
        verdict_str = "SUCCESS" if oracle_verdict else "FAILURE"
        msg = (
            f"\n  Oracle says: {verdict_str} ({n_frames} frames pending). "
            f"ENTER=accept, 'y'=force success, 'n'=force failure: "
        )
        response = self._timed_input(msg).strip().lower()
        if response == "y":
            logging.info("[POST-PLAN] user override → SUCCESS (commit)")
            return True
        if response == "n":
            logging.info("[POST-PLAN] user override → FAILURE (discard)")
            return False
        return oracle_verdict

    def _prompt_human_only_verdict(self, n_frames: int) -> tuple[bool, str]:
        """Ask the human for the verdict directly, with no oracle baseline.

        Used when ``--no-use-oracle`` disables the VLM oracle. There's no
        sensible default here (no oracle verdict to fall back to), so the
        loop keeps re-asking until it gets a clean ``y`` / ``n``. Returns
        ``(committed, short_reason)`` where ``short_reason`` is the bare
        ``SUCCESS`` / ``FAILURE`` tag (the caller prefixes ``human-only:``).
        """
        msg = (
            f"\n  No oracle (--no-use-oracle). {n_frames} frames pending. "
            f"Was this trial a SUCCESS? 'y' = success (commit), "
            f"'n' = failure (discard): "
        )
        while True:
            response = self._timed_input(msg).strip().lower()
            if response == "y":
                logging.info("[POST-PLAN] human verdict → SUCCESS (commit)")
                return True, "SUCCESS"
            if response == "n":
                logging.info("[POST-PLAN] human verdict → FAILURE (discard)")
                return False, "FAILURE"
            print("    Please enter 'y' or 'n'.")
