"""xArm runtime + flywheel package.

Two entry points:

- ``run_flywheel(args: FlywheelArgs)`` — VLM-planned execution with skill gaps.
- ``run_primitives(args: PrimitivesArgs)`` — fixed primitive sequence.

Both share ``XArmHardware`` + ``XArmRunner`` (the trained-policy loop, output
saving, VLM done-checks). Flywheel mode adds planning + skill-gap dispatch
on top via ``XArmFlywheelExecutor``.
"""

from __future__ import annotations

from .args import FlywheelArgs, PrimitivesArgs, RuntimeArgs
from .executor import XArmFlywheelExecutor
from .hardware import XArmHardware
from .main import run, run_flywheel, run_primitives
from .recording import FlywheelRecorder
from .runner import XArmRunner

__all__ = [
    "FlywheelArgs", "PrimitivesArgs", "RuntimeArgs",
    "XArmHardware", "XArmRunner", "XArmFlywheelExecutor",
    "FlywheelRecorder",
    "run", "run_flywheel", "run_primitives",
]
