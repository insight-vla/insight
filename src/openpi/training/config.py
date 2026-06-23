"""Unified openpi training config registry.

This module is a thin shim that:
  * Re-exports the shared dataclasses (TrainConfig, DataConfig, AssetsConfig,
    DataConfigFactory, ModelTransformFactory, and the generic Aloha / LIBERO /
    DROID data configs) from :mod:`openpi.training.sim_configs`.
  * Re-exports the xArm-real data configs from
    :mod:`openpi.training.xarm_configs`.
  * Concatenates ``_SIM_CONFIGS`` and ``_XARM_CONFIGS`` into a single
    ``_CONFIGS`` list, deduplicating by ``TrainConfig.name`` (xArm entries
    take precedence over sim entries when both define the same name — many
    overlapping xArm-task names exist because sim uses
    ``LeRobotXarmPrimDataConfig`` while real uses ``LeRobotXarmDataConfig``;
    keeping the real-hardware variant is the safer default at the unified
    entry point).
  * Provides ``get_config(name)`` and ``cli()`` helpers used by training and
    serving scripts.
"""

from __future__ import annotations

import difflib

import tyro

# Re-export shared dataclasses defined in sim_configs.
from openpi.training.sim_configs import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    FakeDataConfig,
    Filter,
    GroupFactory,
    LeRobotAlohaDataConfig,
    LeRobotDROIDDataConfig,
    LeRobotLegoDataConfig,
    LeRobotLiberoDataConfig,
    LeRobotXarmPrimDataConfig,
    ModelTransformFactory,
    ModelType,
    RLDSDroidDataConfig,
    SimpleDataConfig,
    TrainConfig,
)
from openpi.training.sim_configs import _SIM_CONFIGS

# Re-export xArm-real-only data configs.
from openpi.training.xarm_configs import (
    LeRobotXarmAADataConfig,
    LeRobotXarmDataConfig,
)
from openpi.training.xarm_configs import _XARM_CONFIGS

__all__ = [
    "AssetsConfig",
    "DataConfig",
    "DataConfigFactory",
    "FakeDataConfig",
    "Filter",
    "GroupFactory",
    "LeRobotAlohaDataConfig",
    "LeRobotDROIDDataConfig",
    "LeRobotLegoDataConfig",
    "LeRobotLiberoDataConfig",
    "LeRobotXarmAADataConfig",
    "LeRobotXarmDataConfig",
    "LeRobotXarmPrimDataConfig",
    "ModelTransformFactory",
    "ModelType",
    "RLDSDroidDataConfig",
    "SimpleDataConfig",
    "TrainConfig",
    "cli",
    "get_config",
]


def _merge_configs() -> list[TrainConfig]:
    """Concatenate sim + xArm configs, deduping by name (xArm wins)."""
    merged: dict[str, TrainConfig] = {}
    for cfg in _SIM_CONFIGS:
        merged[cfg.name] = cfg
    # xArm entries overwrite sim entries with the same name. This is the
    # canonical behavior for the unified release: sim and real share many
    # task names (e.g. ``xarm_pick_from_side_50``), but the real-hardware
    # data pipeline is the one we ship by default.
    for cfg in _XARM_CONFIGS:
        merged[cfg.name] = cfg
    return list(merged.values())


_CONFIGS: list[TrainConfig] = _merge_configs()
_CONFIGS_DICT: dict[str, TrainConfig] = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
