"""``XArmFlywheelExecutor`` — flywheel-mode plan-and-execute on the real xArm.

Subclasses ``XArmRunner`` (known-primitive execution + artifact saving) and
mixes in ``_SkillGapMixin`` (skill-gap motion control) and ``_OracleMixin``
(post-plan success oracle + atomic commit/discard). This file contains
only the trial entry point (``run_flywheel``) and the per-step dispatch
(``execute_step``); plumbing-heavy methods live in their dedicated modules:

- ``skill_gap.py`` — pre-analysis + translation servo + rotation planned-motion.
- ``oracle.py`` — return-to-home + visual-diff oracle + commit/discard.

Keep this file focused on the trial control-flow (capture → plan → execute
known + gap primitives → oracle → commit) so the high-level pipeline reads
in one screen.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from insight.planning import log_plan
from insight.reasoning import plan_task

from .oracle import _OracleMixin
from .runner import FrameEntry, XArmRunner, _AbortTrial, _SkipPrimitive
from .skill_gap import _SkillGapMixin, _SkillGapMotion  # noqa: F401  (re-export for legacy callers)

if TYPE_CHECKING:
    from .args import FlywheelArgs


class XArmFlywheelExecutor(_SkillGapMixin, _OracleMixin, XArmRunner):
    """Real-world xArm flywheel: VLM-planned primitive sequence with skill gaps."""

    args: "FlywheelArgs"  # narrowed from RuntimeArgs

    # ────────────────── Public entry ──────────────────

    def run_flywheel(self, dataset_durations: dict[str, int] | None = None) -> bool:
        """Plan + execute end to end. Returns True on success."""
        args = self.args
        self._dataset_durations = dataset_durations or {}
        # All recordings (known primitives + skill gap) are deferred and
        # commit/discard together based on the post-plan oracle verdict.
        # This ensures the saved dataset only contains MUTUALLY consistent
        # episode pairs — e.g., we don't save a move-to-rocks episode whose
        # sweep counterpart was judged a failure.
        self._pending_episodes: list[tuple[str, list[dict]]] = []
        self._had_skill_gap_in_plan: bool = False
        # Tracks whether execute_step has already processed a skill-gap this
        # trial. Used by record_skill_gap_only=False to keep lead-in known
        # primitives (move-to-bottle, close) and drop post-gap ones (lift,
        # open) — pre-gap context defines the new skill, post-gap is just
        # the trained policy's existing behavior.
        self._skill_gap_seen_this_trial: bool = False
        # Most recently executed skill-gap motion in this trial, formatted as a
        # human-readable string (e.g. "'pour' rotation axis=dry by +90.0°").
        # Passed to the next skill gap's pre-analysis so the VLM can reason
        # about whether THIS primitive is the inverse of the prior one (and
        # pick same-axis-opposite-sign accordingly).
        self._last_skill_gap_summary: str | None = None
        # Reset every trial so main.py's per-trial reason capture doesn't leak
        # the previous trial's reason on early-exit paths.
        self._last_trial_reason = ""
        # Per-trial metrics surfaced to main.py for the trials log. Reset here
        # (not in __init__) so each trial starts from a clean slate even if a
        # previous trial bailed before all fields were set.
        self._last_plan_length = 0
        self._last_skill_gaps: list[str] = []
        self._last_episodes_committed = 0
        self._last_frames_committed = 0
        self._last_oracle_overridden = False
        # Per-trial timing buckets. See XArmRunner.__init__ for the full
        # decomposition; reset here so accumulation is per-trial and main.py
        # reads them after run_flywheel returns.
        self._last_prompt_time_s = 0.0
        self._last_vlm_time_s = 0.0
        self._last_oracle_time_s = 0.0
        self._last_return_home_time_s = 0.0

        # Force a return-to-home before any per-trial work. Guarantees every
        # trial starts from the same configuration regardless of where the
        # previous trial's return-to-home left the arm or whether the user
        # nudged it manually between trials. Idempotent if the arm is already
        # at home (set_position no-ops). Timed into _last_return_home_time_s
        # so paper-reported Exec excludes this deterministic-reset motion.
        if not args.dry_run:
            logging.info("[TRIAL-START] driving to canonical home before plan capture...")
            self._timed_return_home(
                self.hardware.home_pose,
                joint_speed_deg_s=args.home_joint_speed_deg_s,
            )

        # Snapshot the home pose + before-frame. Both are now consistent
        # across trials because we just drove to home.
        scene, _ = self._capture_scene()
        self._home_pose = (
            self.hardware.home_pose.copy() if not args.dry_run else None
        )
        self._before_frame = scene

        plan = self._timed_vlm(plan_task, args.goal, scene, list(args.available_primitives), args.scene_context)
        primitive_sequence = plan.get("primitive_sequence") or []
        skill_gaps = set(plan.get("skill_gaps") or [])
        step_notes = plan.get("step_notes") or []
        self._had_skill_gap_in_plan = bool(skill_gaps)
        # Stash for the skill-gap pre-analysis (mirrors sim's prompt plumbing:
        # goal + per-step planner notes are spliced into the preanalyze
        # prompt so user-typed magnitude hints in --goal flow through).
        self._plan_context: list[tuple[str, str]] = list(zip(
            primitive_sequence, step_notes + [""] * (len(primitive_sequence) - len(step_notes)),
        ))

        if not primitive_sequence:
            logging.error("Planner returned empty primitive_sequence; aborting.")
            self._last_trial_reason = "planner returned empty primitive_sequence"
            return False

        self._last_plan_length = len(primitive_sequence)
        self._last_skill_gaps = sorted(skill_gaps)

        logging.info("Plan reasoning: %s", plan.get("reasoning", ""))
        logging.info("Confidence: %s", plan.get("confidence", "?"))
        log_plan(primitive_sequence, skill_gaps, step_notes,
                 header=f"Plan ({len(primitive_sequence)} steps):")

        # Optional plan-confirmation gate. Lets the user reject obviously-bad
        # plans (wrong axis pick, missed step, etc.) before any motion runs.
        # Plan stats above are already captured so the rejected trial still
        # records plan_length / skill_gaps in flywheel_trials.jsonl.
        if self.args.confirm_plan:
            resp = self._timed_input(
                "\n  [CONFIRM PLAN] ENTER=run, n=reject (mark failure + advance), "
                "Ctrl+C=abort batch: "
            ).strip().lower()
            if resp in ("n", "no"):
                logging.info("[PLAN-REJECTED] user marked plan as bad; skipping execution.")
                self._last_trial_reason = "user-rejected plan"
                return False

        self._prepare_and_run(
            primitive_sequence,
            skill_gaps=skill_gaps,
            plan_summary=str(plan),
        )
        # Trial success = oracle confirms the recording was worth keeping.
        # We deliberately ignore the geometric ``ok`` from _prepare_and_run:
        # bound-hit / max-steps-cap / etc. all funnel through the oracle, which
        # decides based on actual scene change rather than motion completion.
        success, reason = self._finalize_skill_gap_recording()
        self._last_trial_reason = reason
        return success

    # ────────────────── Per-step dispatch (known + skill-gap) ──────────────────

    def execute_step(self, primitive: str, num_steps: int,
                     primitive_idx: int, plan_total: int,
                     is_skill_gap: bool) -> bool:
        """Dispatch one plan step.

        Skill gaps go through the P-control loop; known primitives go through
        the inherited inner loop. In both cases we defer the recording commit
        to the post-plan oracle when the plan contains a skill gap, so the
        whole trial's episodes commit or discard together. Without a skill
        gap, known primitives save immediately via the parent class."""
        det = self._maybe_run_deterministic(primitive, primitive_idx, plan_total)
        if det is not None:
            return det
        # Once the trial is marked failed (user rejected a skill-gap), skip
        # all remaining SKILL-GAP primitives but keep running known
        # primitives so the robot can safely lower / release whatever it's
        # holding instead of leaving the arm stranded at the rejection pose.
        if getattr(self, "_trial_failed_skip_gaps", False) and is_skill_gap:
            logging.info(
                "  [POST-REJECT] skipping skill-gap %r (trial already marked failed)",
                primitive,
            )
            return True
        if not is_skill_gap and not self._had_skill_gap_in_plan:
            return super().execute_step(primitive, num_steps, primitive_idx, plan_total,
                                        is_skill_gap=False)

        prim_frames: list[FrameEntry] | None = [] if self.args.save_videos else None
        if self.recorder is not None:
            self.recorder.start_primitive(primitive)
        ok = False
        user_skipped = False
        interrupted = False
        try:
            try:
                if is_skill_gap:
                    ok = self._execute_skill_gap(primitive, prim_frames)
                else:
                    ok = self._run_known_primitive(primitive, num_steps, prim_frames)
            except _SkipPrimitive:
                logging.info("  [USER-SKIP] %r — advancing to next primitive", primitive)
                user_skipped = True
            except _AbortTrial as e:
                # User pressed 'f' at the skill-gap confirm gate. Mark the
                # trial failed and set a flag so subsequent skill-gaps are
                # skipped; downstream known primitives (lower, open) still
                # run so the bottle gets set down safely.
                logging.info(
                    "  [POST-REJECT] trial failed — continuing plan with skill-gaps "
                    "skipped so cleanup primitives can run"
                )
                self._trial_failed_skip_gaps = True
                self._last_trial_reason = f"user-rejected skill-gap: {e}"
                user_skipped = True  # treat as skipped for artifact finalization
                ok = False
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            # Defer recorder commit to the post-plan oracle. Skill-gap path
            # always defers (even geometric failures may represent useful
            # displacement that the oracle should judge). Known primitives:
            # - record_skill_gap_only=True (default): never deferred.
            # - record_skill_gap_only=False: deferred if successful AND we
            #   haven't passed a skill-gap yet this trial. Pre-gap primitives
            #   (move-to-bottle, close) are the lead-in that defines the new
            #   skill's setup; post-gap primitives (lift, open) are already
            #   handled by the trained policy and shouldn't bias retraining.
            keep_known = (
                ok
                and not self.args.record_skill_gap_only
                and not self._skill_gap_seen_this_trial
            )
            should_defer = self.recorder is not None and not interrupted and (
                is_skill_gap or keep_known
            )
            if should_defer:
                self._pending_episodes.append(
                    (primitive, self.recorder.pop_buffer())
                )
            elif self.recorder is not None:
                self.recorder.discard_primitive()
            # Set AFTER the deferral decision so the gap itself defers
            # before this flag flips for downstream primitives.
            if is_skill_gap:
                self._skill_gap_seen_this_trial = True
            self._finalize_primitive_artifacts(
                primitive, primitive_idx, plan_total, prim_frames,
                ok=ok, user_skipped=user_skipped, interrupted=interrupted,
                defer_recorder=True,
                is_skill_gap=is_skill_gap,
            )
        # Always continue the plan — oracle decides true success at the end.
        return True
