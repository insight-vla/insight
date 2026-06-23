"""Aggregators over sequences of VLM verdicts.

The single-call majority vote is already inside ``check_primitive_done_verbose``
(N independent calls on the same frame). This module adds *temporal*
aggregation: requiring multiple consecutive True verdicts across successive
frames before declaring the primitive done. This dramatically reduces
false-positive completions from one-off VLM mistakes, which is the dominant
failure mode the team has observed (e.g. "move gripper to the rocks" returning
True too early on a single noisy frame).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TemporalConsistency:
    """Fire only after ``required`` consecutive True verdicts; any False resets.

    Designed for an inference loop: call ``update(verdict)`` once per VLM check,
    then read ``.fired`` (or check the return value of ``update``) to decide
    whether to advance the primitive. ``required=1`` is a no-op (single-vote
    behavior); ``required=3`` is a reasonable default for fine-grained tasks
    where the VLM occasionally false-positives.
    """

    required: int = 3
    _streak: int = 0

    def __post_init__(self) -> None:
        if self.required < 1:
            raise ValueError(f"TemporalConsistency.required must be >= 1, got {self.required}")

    def update(self, verdict: bool) -> bool:
        if verdict:
            self._streak += 1
        else:
            self._streak = 0
        return self.fired

    @property
    def fired(self) -> bool:
        return self._streak >= self.required

    @property
    def streak(self) -> int:
        return self._streak

    def reset(self) -> None:
        self._streak = 0


def should_check_now(step: int, warmup: int, interval: int) -> bool:
    """True when ``step >= warmup`` and ``step`` lands on the check-interval grid.

    Gates periodic VLM done-checks inside an execution loop: skip checks during
    warmup (when the policy hasn't had time to act yet), then poll every
    ``interval`` steps after that. ``interval <= 0`` is treated as "never".
    """
    if interval <= 0:
        return False
    return step >= warmup and step % interval == 0
