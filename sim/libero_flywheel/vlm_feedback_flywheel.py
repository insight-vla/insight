#!/usr/bin/env python3
"""Entry point for the flywheel pipeline.

Thin wrapper — all logic lives in ``vlm_flywheel.flywheel_execution``.

Usage:
    python sim/libero_flywheel/vlm_feedback_flywheel.py \
        --args.task drawer --args.seed 0 --args.num_runs 50 \
        --args.target_successes 30 --args.vlm gemini --args.record
"""
from vlm_flywheel.flywheel_execution import main, FlywheelArgs  # noqa: F401

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
