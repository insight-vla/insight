"""Env-agnostic dataclasses returned by VLM-reasoning functions.

These hold structured VLM output for callers (sim flywheel, xArm flywheel).
Sim/real-specific containers (e.g. observation snapshots, tyro Args) stay in
their respective packages.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class VLMFeedback:
    """Structured feedback from VLM about primitive execution."""
    success: bool
    confidence: float
    description: str
    error_type: str | None = None
    correction: str | None = None
    next_primitive: str | None = None


@dataclasses.dataclass
class TaskPlan:
    """VLM-generated plan for achieving a goal."""
    goal: str
    primitive_sequence: list[str]
    skill_gaps: list[str]
    reasoning: str
    confidence: float
    requires_new_primitive: bool


@dataclasses.dataclass
class ActionCorrection:
    """VLM-generated action-level correction (7-DOF delta + metadata)."""
    action_delta: list[float]
    description: str
    confidence: float
    should_abort: bool = False
    switch_to_primitive: str | None = None
