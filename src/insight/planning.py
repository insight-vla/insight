"""Helpers for working with VLM-generated plans (env-agnostic).

A "plan" here is the dict returned by ``insight.reasoning.plan_task``:
``{primitive_sequence, step_notes, skill_gaps, ...}``. These helpers do the
boring per-step bookkeeping that both sim and real pipelines need.
"""

from __future__ import annotations

import logging


def format_plan(
    primitive_sequence: list[str],
    skill_gaps: list[str] | set[str],
    step_notes: list[str] | None = None,
    note_width: int = 80,
    name_width: int = 50,
) -> list[str]:
    """Format a plan as a list of human-readable lines.

    Each line looks like::

        [3] [SKILL-GAP] push object left                                  — execute slowly to avoid spilling

    Returns the lines so callers can ``logging.info`` each one (or join with
    newlines for a single block). Doesn't print itself — the caller chooses
    the log level / destination.
    """
    skill_gap_set = set(skill_gaps)
    notes = step_notes or []
    lines: list[str] = []
    for i, p in enumerate(primitive_sequence):
        tag = "SKILL-GAP" if p in skill_gap_set else "known"
        note = notes[i] if i < len(notes) else ""
        lines.append(
            f"  [{i + 1}] [{tag}] {p:<{name_width}} — {note[:note_width]}"
        )
    return lines


def log_plan(
    primitive_sequence: list[str],
    skill_gaps: list[str] | set[str],
    step_notes: list[str] | None = None,
    header: str = "",
    log: logging.Logger | None = None,
) -> None:
    """Convenience: format the plan and log each line at INFO."""
    logger = log or logging.getLogger()
    if header:
        logger.info(header)
    for line in format_plan(primitive_sequence, skill_gaps, step_notes):
        logger.info(line)


def resolve_step_durations(
    primitive_sequence: list[str],
    durations: dict[str, int],
    default: int,
    fixed: int | None = None,
) -> list[tuple[str, int]]:
    """Pair each primitive with its execution-step budget.

    Lookup priority: ``fixed`` (if positive) > per-name in ``durations`` >
    ``default``. Logs a warning when falling back to ``default``. Returns
    ``[(primitive, num_steps), ...]`` so the caller can iterate the plan
    with a known cost per step.
    """
    out: list[tuple[str, int]] = []
    for p in primitive_sequence:
        if fixed is not None and fixed > 0:
            dur = fixed
        elif p in durations:
            dur = durations[p]
        else:
            logging.warning("No duration for %r; using default=%d", p, default)
            dur = default
        out.append((p, dur))
    return out
