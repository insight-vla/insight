"""Abstract executor base class for VLM-driven plan execution.

The InSight pipeline (sim flywheel + real-world xArm) executes plans by
dispatching primitives to one of two paths:

- Known primitives → trained VLA policy (env-specific action interface)
- Skill gaps      → VLM-driven action generation OR P-control toward a target

Concrete subclasses provide the env-specific bits (``execute_step``, the
``run_*`` entry methods). The env-agnostic VLM helpers
(``evaluate_progress``, ``generate_action``) are implemented here in terms of
``insight.reasoning`` and need no per-env override unless a subclass
wants different behavior — which is exactly what flywheel-mode subclasses do.

Design notes:

- The ``GENERATE_PROMPT`` class attribute lets subclasses pick a prompt variant
  (e.g. flywheel mode uses an axis-aware variant) without overriding the
  method body. Per-instance shadowing (``self.GENERATE_PROMPT = ...``) is
  supported for runtime feedback injection.
- ``args`` is held on the instance so methods can access run configuration
  without it being threaded through every call.
- Subclasses MUST implement ``execute_step`` and one or more ``run_*`` entry
  methods. The base class deliberately omits a top-level ``run`` entry so
  each subclass can name it for its mode (e.g. ``run_adaptive``,
  ``run_flywheel_adaptive``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from insight.prompts import GENERATE_NEW_PRIMITIVE_SYSTEM
from insight.reasoning import (
    evaluate_new_primitive_progress,
    generate_new_primitive_action,
)


class BaseExecutor(ABC):
    """Abstract base for plan-and-execute pipelines.

    Concrete subclasses (``LiberoSimExecutor``, ``XArmExecutor`` and their
    flywheel variants) bind the env-specific bits and override the prompt /
    method swappables that change between adaptive and flywheel modes.
    """

    # Subclasses override to select a prompt variant (e.g. flywheel mode).
    GENERATE_PROMPT: str = GENERATE_NEW_PRIMITIVE_SYSTEM

    def __init__(self, args: Any) -> None:
        self.args = args

    # ────────────────── Env-agnostic VLM behaviors ──────────────────

    def evaluate_progress(
        self,
        initial_img,
        current_img,
        primitive: str,
        goal: str,
        steps: int,
    ) -> tuple[bool, str]:
        """Per-step progress check during a skill-gap (new primitive) loop.

        Default delegates to ``insight.reasoning.evaluate_new_primitive_progress``.
        Flywheel subclasses override to use sim ground-truth (peg direction) or
        a different VLM variant.
        """
        return evaluate_new_primitive_progress(
            initial_img, current_img, primitive, goal, steps,
        )

    def generate_action(
        self,
        initial_img,
        current_img,
        wrist_img,
        primitive: str,
        step: int,
        goal: str = "",
        feedback: str = "",
    ) -> tuple:
        """VLM action generator for skill-gap primitives.

        Returns ``(action, description, done)``. Uses ``self.GENERATE_PROMPT``
        so subclasses can swap the prompt variant via class attribute (no
        method override needed).
        """
        return generate_new_primitive_action(
            initial_img, current_img, wrist_img, primitive, step,
            goal=goal, feedback=feedback, system=self.GENERATE_PROMPT,
        )

    # ────────────────── Env-specific (subclass implements) ──────────────────

    @abstractmethod
    def execute_step(self, *args, **kwargs) -> Any:
        """Dispatch a single plan step. Concrete subclasses know how to drive
        their environment's action interface (LIBERO ``env.step``, xArm
        ``arm.set_servo_cartesian``, etc.) and how to handle skill gaps."""
