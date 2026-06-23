"""Real-hardware xArm training configs.

This module defines xArm-real-only data configs (LeRobotXarmDataConfig,
LeRobotXarmAADataConfig) and the list of xArm-real registered configs
``_XARM_CONFIGS``. All shared dataclasses (TrainConfig, DataConfig,
AssetsConfig, DataConfigFactory, ModelTransformFactory, Aloha/LIBERO/DROID
configs, etc.) are imported from :mod:`openpi.training.sim_configs`.

The thin shim :mod:`openpi.training.config` concatenates ``_SIM_CONFIGS`` +
``_XARM_CONFIGS`` and re-exports the shared dataclasses.
"""

import dataclasses
import pathlib
from typing import TypeAlias

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.policies.droid_policy as droid_policy
import openpi.policies.xarm_policy as xarm_policy
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# Shared dataclasses live in sim_configs.py — import the ones used below.
from openpi.training.sim_configs import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    FakeDataConfig,
    LeRobotAlohaDataConfig,
    LeRobotDROIDDataConfig,
    LeRobotLiberoDataConfig,
    ModelTransformFactory,
    RLDSDroidDataConfig,
    SimpleDataConfig,
    TrainConfig,
)

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class LeRobotXarmDataConfig(DataConfigFactory):
    """
    Example data config for custom Xarm dataset in LeRobot format for primitives dataset.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/state": "state",
                        "actions": "actions",
                        # Use the LeRobotDataset's ``task`` field directly as the
                        # prompt (it already contains the task description string).
                        # This lets configs set ``prompt_from_task=False`` without
                        # needing the PromptFromLeRobotTask transform to populate
                        # the "prompt" key.
                        "prompt": "task",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, 0)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        
        # no delta transforms
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotXarmAADataConfig(DataConfigFactory):
    """
    Example data config for custom Xarm dataset in LeRobot format for primitives dataset.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )
        
        # no delta transforms
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

# Use `get_config` if you need to get a config by name in your code.
_XARM_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # xarm 
    #
    TrainConfig(
        name="xarm_scoop",
        model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
        data=LeRobotXarmDataConfig(
            # Replace with your custom Xarm LeRobot dataset repo id.
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Comput norm stats of the dataset using-> uv run scripts/compute_norm_stats.py --config-name pi05_xarm_finetune
                # Then possibly use those norm stats and change below
                assets_dir="/home/xarm/openpi/checkpoints/xarm_scoop/scoop_v2/220001/assets", # this might not be necessary
                asset_id="maggiewang/xarm_scoop_75_04_15", # for norm stats (inference and training)
            ),
        ),
    ),
    # TrainConfig(
    #     name="xarm_sweep",
    #     model=pi0_config.Pi0Config(action_horizon=50, pi05=True),
    #     data=LeRobotXarmDataConfig(
    #         # Replace with your custom Xarm LeRobot dataset repo id.
    #         base_config=DataConfig(prompt_from_task=True),
    #         assets=AssetsConfig(
    #             assets_dir="/home/xarm/openpi/checkpoints/xarm_sweep_100_04_20/19999/assets", # this might not be necessary
    #             asset_id="maggiewang/xarm_sweep_100_04_17", # for norm stats (inference and training)
    #         ),
    #     ),
    # ),

    # Pi0.5 xArm sweep (100 full-task demos, end-to-end, no primitive segmentation)
    TrainConfig(
        name="xarm_sweep",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # data=LeRobotXarmPrimDataConfig(
        data=LeRobotXarmDataConfig(
            repo_id="maggiewang/xarm_sweep_100_04_17",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="maggiewang/xarm_sweep_100_04_17"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm scoop (end-to-end, 100 demos, action_horizon=10)
    # Norm stats expected at <ckpt>/assets/maggiewang/xarm_scoop_100_04_19/norm_stats.json.
    TrainConfig(
        name="xarm_scoop_100_ah10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggiewang/xarm_scoop_100_04_19",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="maggiewang/xarm_scoop_100_04_19"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm scoop (primitive-conditioned, 100 demos, action_horizon=10)
    # Norm stats expected at <ckpt>/assets/maggiewang/xarm_scoop_100_primitives_trimmed/norm_stats.json.
    # Primitive prompts: "move gripper to the rocks", "scoop the rocks", "lift upward".
    # Training data is 10Hz — inference loop must match.
    TrainConfig(
        name="xarm_scoop_primitives_ah10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggiewang/xarm_scoop_100_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="maggiewang/xarm_scoop_100_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # xArm pickplace top-grasp primitives. 30 demos × 5 primitives = 160 episodes.
    # ah=10 (matches sweep — long horizons over-smooth contact). 8D actions
    # (6 pose + gripper + progress). prompt_from_task=False so primitive-level
    # training matches xarm_scoop_100_primitives.
    TrainConfig(
        name="xarm_pick_up_top_30_primitives_05_01",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_up_top_30_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_up_top_30_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # V2 dataset (delta-based gripper trim, removes ~89% of close-gripper
    # "still open" prefix). Two horizons trained in parallel for comparison.
    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah10",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm pickplace -> twist: pickplace v2 + flywheel-collected twist
    # primitive (30 demos, normalized from 13 VLM phrasings to "twist open
    # the cap"). Trained on oskd-side, served here. ah=10 to match v2_ah10.
    TrainConfig(
        name="xarm_pickplace_to_twist_30",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pickplace_to_twist_30",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_to_twist_30"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm pickplace + flywheel-acquired twist (v4): top-grasp pickplace
    # base + 20 flywheel-collected twist demos label-normalized to canonical.
    TrainConfig(
        name="xarm_pickplace_to_twist_v4",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pickplace_to_twist_v4",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_to_twist_v4"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm unified skills (2026-05-10): top + twist + side + pour with
    # move-to-bottle renamed per grasp type. Single policy handles both
    # twist (top-grasp) and pour (side-grasp) tasks.
    TrainConfig(
        name="xarm_unified_skills_05_10",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_unified_skills_05_10",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_unified_skills_05_10"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm unified skills (relabeled 2026-05-11): same source episodes
    # as 05_10 but move-to-bottle renamed to "...the bottle cap"/"...body"
    # (distinct head nouns) instead of "...from the top"/"...from the side"
    # (suffix-only). Distinct head nouns produce larger pooled-embedding
    # distance, giving the policy actual gradient signal to disambiguate.
    TrainConfig(
        name="xarm_unified_skills_05_11",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_unified_skills_05_11",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_unified_skills_05_11"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm unified skills 05_12: same recipe as 05_11 (same RENAMES,
    # same merge) but v5 sources for the top-grasp pickplace + twist:
    #   - PICKPLACE_TOP: xarm_pick_from_top_v4 -> v5 (bowl re-positioned to
    #     match the pour/twist eval scene; same hparams).
    #   - TWIST_CLEAN:   xarm_twist_open_v4_clean_unwrap -> v5_clean_unwrap
    #     (fresh 20 flywheel demos collected against the v5 top policy).
    # Side-grasp + pour sources unchanged from 05_11.
    TrainConfig(
        name="xarm_unified_skills_05_12",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_unified_skills_05_12",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_unified_skills_05_12"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace + flywheel-acquired pour (2026-05-10):
    # side-grasp pickplace v5 base + 20 flywheel pour trials (40 episodes
    # spanning tilt-forward to pour + tilt-back upright).
    TrainConfig(
        name="xarm_pickplace_pour_05_10",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pickplace_pour_05_10",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_pour_05_10"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Same as xarm_pickplace_to_twist_v4 but with rpy unwrap applied to twist
    # data before merging (fixes the 2π discontinuity that caused elevated loss).
    TrainConfig(
        name="xarm_pickplace_to_twist_v4_unwrap",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pickplace_to_twist_v4_unwrap",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_to_twist_v4_unwrap"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # v5 successor to xarm_pickplace_to_twist_v4_unwrap: same recipe (rpy
    # unwrap applied to twist data before merging), but flywheel-collected
    # against the xarm_pick_from_top_v5 policy with the matched v5 pickplace
    # base. Re-collection moved the bowl to match the pour/twist eval scene.
    TrainConfig(
        name="xarm_pickplace_to_twist_v5_unwrap",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pickplace_to_twist_v5_unwrap",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_to_twist_v5_unwrap"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace: 50 human teleop demos of grasping the
    # yellow bottle from the side and placing next to a bowl. Base policy for
    # the pour-primitive flywheel. Trained on oskd-side, served here. ah=10.
    TrainConfig(
        name="xarm_pick_from_side_50",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_side_50_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_side_50_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace, cleaned re-collection: 50 teleop demos
    # with the v2-prompt segmentation pipeline (kinematic chain-of-thought +
    # dom-axis caption + missing-primitives filter). Same architecture as
    # xarm_pick_from_side_50; mirrored from oskd config for hardware serving.
    TrainConfig(
        name="xarm_pick_from_side_clean",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_side_clean_v2_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_side_clean_v2_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace v3: 50 teleop demos re-collected with the
    # workspace repositioned to keep joint 6 away from its angle limit during
    # pour rotations. Trained on oskd-side, served here. Same architecture as
    # xarm_pick_from_side_clean; only raw recording + dataset name differ.
    TrainConfig(
        name="xarm_pick_from_side_v3",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_side_v3_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_side_v3_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm top-grasp pickplace v4: 50 teleop demos. Base policy for the
    # twist-primitive flywheel. Trained on the openpi VLA codebase, served via training/serve_policy.py.
    TrainConfig(
        name="xarm_pick_from_top_v4",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_top_v4_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_top_v4_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm top-grasp pickplace v5: 50 teleop demos re-collected with the
    # bowl matched to the pour/twist eval scene (v4 had the bowl slightly
    # forward, giving the unified policy a confounding visual cue). Same
    # hparams as v4 for a clean A/B.
    TrainConfig(
        name="xarm_pick_from_top_v5",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_top_v5_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_top_v5_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace v4: 50 re-collected teleop demos.
    # Same architecture as v3; raw recording differs. Trained on the openpi VLA codebase, served via training/serve_policy.py.
    TrainConfig(
        name="xarm_pick_from_side_v4",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_side_v4_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_side_v4_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm side-grasp pickplace v5: clean re-collection that avoids bad gimbal-lock
    # angles in the late approach. v4 demos contained operator wrist wiggle near pitch≈90°
    # which the policy learned and replayed at inference, flicking the bottle on grasp.
    # v5 demos keep the gripper away from the singular pose during the final approach.
    # Same architecture as v3/v4.
    TrainConfig(
        name="xarm_pick_from_side_v5",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_from_side_v5_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_side_v5_primitives_trimmed"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah20",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah30",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=30,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_up_top_30_primitives_trimmed_v2"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=30,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Scoop -> sweep: existing scoop primitives + flywheel-bootstrapped sweep,
    # balanced 50/task. Validates whether the existing scoop policy can absorb
    # a new sweep primitive without losing the originals.
    TrainConfig(
        name="xarm_scoop_to_sweep_50_04_28",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmDataConfig(
            repo_id="maggie/xarm_scoop_to_sweep_50_04_28",
            # prompt_from_task=False: the LeRobot ``task`` field is mapped to
            # ``prompt`` directly via the repack transform above.
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_scoop_to_sweep_50_04_28"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 200 episodes (vs 303 in the scoop-only config @ 30k steps).
        # Proportional step count: 30k * (200/303) ≈ 20k.
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see upstream openpi (Physical-Intelligence/openpi) examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see upstream openpi examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs — upstream openpi DROID baselines. Not used
    # by InSight paper experiments. Disabled because the configs reference an
    # older RLDSDroidDataConfig API; uncomment + sync to re-enable.
    # *roboarena_config.get_roboarena_configs(),
    # *polaris_config.get_polaris_configs(),
]

