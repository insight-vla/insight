"""Simulation (LIBERO / LEGO / xArm-primitive-sim) training configs.

This module defines the SHARED dataclasses (TrainConfig, DataConfig, AssetsConfig,
DataConfigFactory, ModelTransformFactory, and the generic Aloha/LIBERO/DROID
data configs) plus the sim-specific configs (LeRobotLegoDataConfig,
LeRobotXarmPrimDataConfig). The list of registered sim configs is
``_SIM_SIM_CONFIGS``.

``training/config.py`` is a thin shim that re-exports the shared classes from
here and concatenates _SIM_SIM_CONFIGS with the real-hardware _XARM_SIM_CONFIGS from
``xarm_configs.py``.
"""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.lego_policy as lego_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.xarm_policy_prim as xarm_policy_prim
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "task",  # Use task field directly (contains the prompt string)
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLegoDataConfig(DataConfigFactory):
    """
    Config for training on LEGO pick and place dataset (LeRobot format).
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform to match dataset keys to policy inference keys
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        # Data transforms using LEGO-specific policy
        data_transforms = _transforms.Group(
            inputs=[lego_policy.LegoInputs(model_type=model_config.model_type)],
            outputs=[lego_policy.LegoOutputs()],
        )

        # Model transforms (tokenization, etc.)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXarmPrimDataConfig(DataConfigFactory):
    """xArm dataset in LeRobot format (full-task demos or primitive segments).

    Name is legacy — works for both full-task demos (e.g. scoop) and
    primitive-segmented data. The key contract is the xarm observation schema
    (two exterior cameras + wrist camera + state).
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
                        "prompt": "task",  # LeRobot stores per-frame label in "task"; remap to "prompt"
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[xarm_policy_prim.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy_prim.XarmOutputs(model_type=model_config.model_type)],
        )

        # First 6 dims (xyz + rpy) are delta actions, last dim (gripper) is absolute.
        delta_action_mask = _transforms.make_bool_mask(6, 0)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see upstream openpi examples/droid/convert_droid_data_to_lerobot.py
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
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_SIM_CONFIGS = [
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
    # Pi0.5 base model tested on LIBERO (uses LIBERO norm stats which have quantiles)
    TrainConfig(
        name="pi05_libero_base_norm",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(asset_id="physical-intelligence/libero"),  # Use LIBERO norm stats (have quantiles)
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see upstream openpi (Physical-Intelligence/openpi) examples/aloha_real/README.md
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
    # #
    # # LEGO pick and place config.
    # #
    # TrainConfig(
    #     name="pi05_lego",
    #     model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    #     data=LeRobotLegoDataConfig(
    #         repo_id="maggiewang/lego_demos",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=64,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=1_000,
    #         peak_lr=5e-5,
    #         decay_steps=30_000,
    #         decay_lr=5e-5,
    #     ),
    #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    #     ema_decay=0.999,
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=10_000,
    # ),
    # TrainConfig(
    #     name="pi05_lego_lora",
    #     model=pi0_config.Pi0Config(
    #         pi05=True,
    #         action_horizon=10,
    #         discrete_state_input=False,
    #         paligemma_variant="gemma_2b_lora",  # Use LoRA on main LLM
    #         action_expert_variant="gemma_300m",  # No LoRA on action expert (keep frozen)
    #     ),
    #     data=LeRobotLegoDataConfig(
    #         repo_id="maggiewang/lego_demos",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=16,  # Reduced from 64 to fit in 24GB GPU with LoRA
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=1_000,
    #         peak_lr=5e-5,
    #         decay_steps=30_000,
    #         decay_lr=5e-5,
    #     ),
    #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=10_000,
    #     # Custom freeze filter: freeze VIT + Action Expert, only train LoRA weights on main LLM
    #     # NOTE: VIT is stored under PaliGemma/img (not siglip) - use .*PaliGemma/img.* to freeze it
    #     freeze_filter=nnx.Any(
    #         # Freeze VIT (stored under PaliGemma/img, not siglip)
    #         nnx_utils.PathRegex(".*PaliGemma/img.*"),
    #         # Freeze Action Expert (flow policy) - all params matching llm.*_1
    #         nnx_utils.PathRegex(".*llm.*_1.*"),
    #         # Freeze main LLM non-LoRA params - LLM params that don't have lora in path
    #         nnx.All(
    #             nnx_utils.PathRegex(".*llm.*"),  # Is an LLM param
    #             nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),  # Not action expert
    #             nnx.Not(nnx_utils.PathRegex(".*lora.*")),  # Not a LoRA weight
    #         ),
    #     ),
    #     # Turn off EMA for LoRA fine-tuning
    #     ema_decay=None,
    # ),
    # TrainConfig(
    #     name="pi05_libero_with_lego",
    #     model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=DataConfig(prompt_from_task=True),
    #         extra_delta_transform=False,
    #     ),
    #     batch_size=256,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000,
    #         peak_lr=5e-5,
    #         decay_steps=1_000_000,
    #         decay_lr=5e-5,
    #     ),
    #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    #     ema_decay=0.999,
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=30_000,
    # ),
    TrainConfig(
        name="pi05_libero_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA
    ),
    # TrainConfig(
    #     name="pi05_libero_lego_lora",
    #     model=pi0_config.Pi0Config(
    #         pi05=True,
    #         action_horizon=10,
    #         discrete_state_input=False,
    #         paligemma_variant="gemma_2b_lora",  # Use LoRA on main LLM
    #         action_expert_variant="gemma_300m",  # No LoRA on action expert (keep frozen)
    #     ),
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="maggiewang/libero_with_lego",
    #         base_config=DataConfig(prompt_from_task=True),
    #         extra_delta_transform=False,
    #     ),
    #     batch_size=16,  # Reduced from 256 to fit in 24GB GPU with LoRA
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000,
    #         peak_lr=5e-5,
    #         decay_steps=1_000_000,
    #         decay_lr=5e-5,
    #     ),
    #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=30_000,
    #     # Custom freeze filter: freeze VIT + Action Expert, only train LoRA weights on main LLM
    #     # NOTE: VIT is stored under PaliGemma/img (not siglip) - use .*PaliGemma/img.* to freeze it
    #     freeze_filter=nnx.Any(
    #         nnx_utils.PathRegex(".*PaliGemma/img.*"),  # Freeze VIT (stored under PaliGemma/img, not siglip)
    #         nnx_utils.PathRegex(".*llm.*_1.*"),  # Freeze Action Expert (flow policy)
    #         nnx.All(
    #             nnx_utils.PathRegex(".*llm.*"),
    #             nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    #             nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    #         ),
    #     ),
    #     ema_decay=None,  # Turn off EMA for LoRA
    # ),


    # Pi0.5 with primitives (mixed with task-level prompts)
    # LoRA on both LLM and action expert, VIT gets trained
    TrainConfig(
        name="pi05_libero_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero,maggiewang/libero_primitives,maggiewang/libero_primitives_part2",
            assets=AssetsConfig(asset_id="physical-intelligence/libero"),
            base_config=DataConfig(prompt_from_task=False),  # Use task field directly via repack
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Pi0.5 with primitives ONLY (no task-level prompts)
    # Train only on primitive-labeled data to maximize steerability
    TrainConfig(
        name="pi05_libero_primitives_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            # ONLY primitives data - no task-level prompts to avoid conflicting signals
            repo_id="maggiewang/libero_primitives,maggiewang/libero_primitives_part2",
            # asset_id must match first repo_id for norm stats to be found
            assets=AssetsConfig(asset_id="maggiewang/libero_primitives"),
            base_config=DataConfig(prompt_from_task=False),  # Use task field directly via repack
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Pi0.5 for lego primitives steerability experiment
    # Small dataset (24 demos) with red/blue lego blocks
    TrainConfig(
        name="pi05_lego_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_primitives_labeled",
            assets=AssetsConfig(asset_id="maggiewang/lego_primitives_labeled"),
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Pi0.5 for lego primitives with dense labels + play data
    TrainConfig(
        name="pi05_lego_primitives_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_vlm_primitives,maggiewang/lego_primitives_play",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_vlm_primitives"),
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Pi0.5 for lego primitives v3 - includes new 150 demos (75 red + 75 blue) from 01/26
    TrainConfig(
        name="pi05_lego_primitives_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_vlm_primitives_01_26,maggiewang/lego_primitives_play",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_vlm_primitives_01_26"),
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 for lego primitives v4 - includes new 150 demos (75 red + 75 blue) from 01/26
    # Try without play data just to see
    TrainConfig(
        name="pi05_lego_primitives_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_vlm_primitives_01_26",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_vlm_primitives_01_26"),
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Pi0.5 for lego primitives v5 - re-converted with correct image flip (180° rotation)
    TrainConfig(
        name="pi05_lego_primitives_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_primitives_v5_01_30_trimmed,maggiewang/lego_primitives_random_fixedflip_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_primitives_v5_01_30_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 for lego primitives v6 - re-converted with correct image flip (180° rotation)
    # no play data, 100 pick and place red and 100 pick and place blue
    # for training on cluster
    TrainConfig(
        name="pi05_lego_primitives_v6",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_primitives_v6_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_primitives_v6_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 lego v7 - 50 red + 50 blue pick-place + 29 flip demos (move-to + rotate block)
    TrainConfig(
        name="pi05_lego_primitives_v7_with_flip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_primitives_v7_with_flip",
            assets=AssetsConfig(asset_id="maggiewang/lego_primitives_v7_with_flip"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 lego v8 - 50 red + 50 blue pick-place + 29 old flip + 30 new flip (P-control corrected)
    TrainConfig(
        name="pi05_lego_primitives_v8_with_flip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_primitives_v8_with_flip",
            assets=AssetsConfig(asset_id="maggiewang/lego_primitives_v8_with_flip"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 oracle flip (merged) - 50 red + 50 blue pick-place + 60 human teleop flip demos
    TrainConfig(
        name="pi05_lego_oracle_flip_03_14",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_oracle_flip_03_14_merged",
            assets=AssetsConfig(asset_id="maggiewang/lego_oracle_flip_03_14_merged"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=32,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 oracle flip ONLY - 140 human teleop flip demos (60 old + 80 new), trimmed idle frames
    TrainConfig(
        name="pi05_lego_oracle_flip_140_03_23",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_oracle_flip_140_03_23_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_oracle_flip_140_03_23_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 tilted pick-place primitives - 100 demos (block starts tilted on side)
    TrainConfig(
        name="pi05_lego_pickplace_tilted_100_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_tilted_100_primitives_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_tilted_100_primitives_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 tilted pick-place primitives - 150 demos (100 + 50 hard orientations)
    TrainConfig(
        name="pi05_lego_pickplace_tilted_150_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_tilted_150_primitives_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_tilted_150_primitives_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 combined pick-place + flywheel flip primitives (150 pick-place + 59 flip)
    TrainConfig(
        name="pi05_lego_pickplace_flip_combined",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_flip_combined_04_02",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_flip_combined_04_02"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 combined pick-place + flywheel flip 246 (150 pick-place + 246 rotate block from flywheel)
    TrainConfig(
        name="pi05_lego_pickplace_flywheel_flip_04_05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_flywheel_flip_04_05",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_flywheel_flip_04_05"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 combined pick-place + flywheel flip 153 (150 pick-place + 153 rotate block from flywheel)
    TrainConfig(
        name="pi05_lego_pickplace_flywheel_flip_04_04",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_flywheel_flip_04_04",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_flywheel_flip_04_04"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 combined pick-place + human flip (150 pick-place + 140 rotate block from human demos)
    TrainConfig(
        name="pi05_lego_human_pickplace_flip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_human_pickplace_flip_04_03",
            assets=AssetsConfig(asset_id="maggiewang/lego_human_pickplace_flip_04_03"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 tilted pick-place e2e - 100 demos, single prompt (no primitives)
    TrainConfig(
        name="pi05_lego_pickplace_tilted_100_e2e",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_pickplace_tilted_100_03_31_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_pickplace_tilted_100_03_31_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 oracle flip primitives - 140 flip demos segmented into primitives
    TrainConfig(
        name="pi05_lego_oracle_flip_140_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_oracle_flip_140_primitives_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_oracle_flip_140_primitives_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 red pick-place end-to-end (single prompt, no primitive segmentation)
    # Control experiment: does single-prompt training fail for pick-place too?
    TrainConfig(
        name="pi05_lego_red_pickplace_e2e",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_red_pickplace_e2e_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/lego_red_pickplace_e2e_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 drawer open top primitives (50 demos, 3 primitives: move to handle, close gripper, pull open)
    TrainConfig(
        name="pi05_drawer_open_top_primitives",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/drawer_open_top_50_04_07_primitives_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/drawer_open_top_50_04_07_primitives_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 drawer open top primitives with progress prediction (8D actions: 7 + progress 0→1)
    TrainConfig(
        name="pi05_drawer_open_top_primitives_progress",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/drawer_open_top_50_04_09_primitives_progress_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/drawer_open_top_50_04_09_primitives_progress_trimmed"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 drawer open + flywheel push (50 open demos + 70 flywheel push demos)
    TrainConfig(
        name="pi05_drawer_open_push_combined_04_12",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/drawer_open_top_50_04_07_primitives_trimmed,maggiewang/drawer_flywheel_push_70_04_12_trimmed",
            assets=AssetsConfig(asset_id="maggiewang/drawer_open_push_combined_04_12"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=64,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 xArm scoop (75 full-task demos — not yet segmented into primitives)
    TrainConfig(
        name="xarm_scoop",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
            repo_id="maggiewang/xarm_scoop_75_04_15",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="maggiewang/xarm_scoop_75_04_15"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=8_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        save_interval=2_000,
        keep_period=6_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm sweep (100 full-task demos, end-to-end, no primitive segmentation)
    # ah=10 validated 5/5 on hardware (2026-04-21); longer chunks over-smooth contact.
    TrainConfig(
        name="xarm_sweep_100_04_20",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # xArm scoop -> sweep: existing scoop primitives (move/scoop/lift) + new
    # flywheel-bootstrapped sweep primitive, balanced 50/task. Validates whether
    # a policy with the existing scoop primitives can absorb a new sweep primitive
    # without losing the originals. Trained from pi05 base on the merged dataset
    # at training/preprocess/ (200 episodes, ~18k frames,
    # 6D actions, fps=10 — sweep timestamps relabeled from 20fps to match scoop).
    TrainConfig(
        name="xarm_scoop_to_sweep_50_04_28",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
            repo_id="maggie/xarm_scoop_to_sweep_50_04_28",
            base_config=DataConfig(prompt_from_task=True),
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
        # 200 episodes, ~18k frames. Same epoch count as scoop-only at 30k steps
        # (which ran on ~303 episodes): 30k * 200/303 ≈ 20k.
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm scoop primitives (100 demos, VLM-labeled into move/scoop/lift, trimmed settling zeros)
    TrainConfig(
        name="xarm_scoop_100_primitives_04_20",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
            repo_id="maggiewang/xarm_scoop_100_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
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
            action_horizon=50,
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

    # Pi0.5 xArm pickplace top-grasp primitives. 30 demos × 5 primitives = 160
    # episodes. ah=10 (matches sweep — long horizons over-smooth contact).
    # 8D actions (6 pose + gripper + progress). prompt_from_task=False so
    # primitive-level training matches xarm_scoop_100_primitives_04_20.
    TrainConfig(
        name="xarm_pick_up_top_30_primitives_05_01",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
        # 160 episodes × ~78 mean frames = ~12.5k frames; scoop-100 used 30k steps on 303 episodes,
        # so scale: 30k × 160/303 ≈ 16k. Round up for safety.
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # V2 dataset (delta-based gripper trim removes 89% of close-gripper "still open"
    # prefix that previously biased the model). Two horizons trained in parallel
    # so we can pick the better-performing one. Both use prompt_from_task=False;
    # primitive labels reach the model via the LeRobotXarmPrimDataConfig repack.
    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah20",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Pi0.5 xArm pickplace -> twist: top-grasp pick-and-place primitives + the
    # flywheel-bootstrapped twist primitive. Validates whether a policy that
    # already knows top-grasp pickplace can absorb a NEW twist primitive
    # without losing the originals. Trained from pi05 base on the merged
    # dataset at training/preprocess/merge_pickplace_twist.py (~180 episodes:
    # 150 pickplace + 30 normalized twist, 6 task labels, 8D actions
    # (pose+gripper+progress, twist columns synthesized in
    # filter_normalize_twist.py), fps=20). ah=10 matches the action-horizon
    # used at deployment for scoop and the production pickplace policy;
    # ah=20/30 sweeps were exploratory.
    # Pi0.5 xArm top-grasp pickplace v2 (50 demos): expanded base policy
    # using the new hybrid segmentation pipeline (velocity-based gripper
    # detection + sub-window VLM). Successor to
    # xarm_pick_up_top_30_primitives_v2_ah10 which used the older
    # threshold-based segmenter on 30 demos. Source repo:
    # maggie/xarm_pick_from_top_v2 (mirrors xarm_pick_from_side naming).
    TrainConfig(
        name="xarm_pick_from_top_v2_50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
            repo_id="maggie/xarm_pick_from_top_v2_50_primitives_trimmed",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pick_from_top_v2_50_primitives_trimmed"),
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
    # the pour-primitive flywheel — pour will be acquired on top of this via
    # a side-grasp + tilt motion. Segmented via the hybrid action-anchor +
    # sub-window VLM pipeline (densely_label_dataset.py), trimmed via
    # preprocess_all_primitives.py.
    TrainConfig(
        name="xarm_pick_from_side_50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # with reduced settling and tighter primitive boundaries vs xarm_pick_from_side_50.
    # Same architecture, same data pipeline (densely_label + preprocess_all_primitives).
    TrainConfig(
        name="xarm_pick_from_side_clean",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # pour rotations (clean_v2 left the gripper at yaw≈170° at end of move-to-
    # bowl, which crashed pour with ControllerError 16/servo-23). Same pipeline,
    # same architecture; only the raw recording differs.
    TrainConfig(
        name="xarm_pick_from_side_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # twist-primitive flywheel.
    TrainConfig(
        name="xarm_pick_from_top_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # bowl positioned to match the pour/twist eval scene (v4's bowl was slightly
    # forward, giving the unified policy a visual cue to default to side-grasp).
    # Same hparams as v4 for a clean A/B.
    TrainConfig(
        name="xarm_pick_from_top_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # Same architecture and pipeline as v3; raw recording differs.
    TrainConfig(
        name="xarm_pick_from_side_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
        name="xarm_pickplace_to_twist_30",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
            repo_id="maggie/xarm_pickplace_to_twist_30",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(asset_id="maggie/xarm_pickplace_to_twist_30"),
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
        # ~180 episodes; matches the scoop_to_sweep epoch count at 20k steps.
        num_train_steps=20_000,
        save_interval=5_000,
        keep_period=5_000,
        batch_size=64,
        wandb_enabled=True,
    ),

    # Pi0.5 xArm pickplace + flywheel-acquired twist (v4): top-grasp pickplace
    # base (xarm_pick_from_top_v4_primitives_trimmed) + 20 flywheel-collected
    # twist demos label-normalized to "twist open the cap". The flywheel
    # validation: can the policy absorb a new bootstrapped primitive without
    # losing the originals?
    TrainConfig(
        name="xarm_pickplace_to_twist_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Pi0.5 xArm unified skills (2026-05-10): top-grasp pickplace v4 +
    # flywheel twist + side-grasp pickplace v5 + flywheel pour, merged with
    # move-to-bottle renamed per grasp type ("from the top" / "from the
    # side"). A single policy trained on this dataset handles BOTH twist
    # tasks (via top-grasp) and pour tasks (via side-grasp); the planner
    # disambiguates by emitting the correct move-to-bottle variant given
    # task context.
    TrainConfig(
        name="xarm_unified_skills_05_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
    # as xarm_unified_skills_05_10 but with stronger label disambiguation —
    # move-to-bottle renamed to "move gripper to the bottle cap" (top) and
    # "move gripper to the bottle body" (side). Distinct head nouns instead
    # of suffix-only "from the top"/"from the side", which had too-similar
    # pooled language embeddings for the policy to disambiguate at 10k
    # steps. Same model + optimizer hparams as 05_10 for a clean A/B.
    TrainConfig(
        name="xarm_unified_skills_05_11",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Pi0.5 xArm unified skills 05_12: same recipe + label conventions as
    # 05_11, but v5 sources for the top-grasp pickplace + twist:
    #   - PICKPLACE_TOP: xarm_pick_from_top_v4 -> v5 (bowl re-positioned to
    #     match the pour/twist eval scene; same hparams).
    #   - TWIST_CLEAN:   xarm_twist_open_v4_clean_unwrap -> v5_clean_unwrap
    #     (fresh 20 flywheel demos collected against the v5 top policy).
    # Side-grasp + pour sources unchanged from 05_11.
    TrainConfig(
        name="xarm_unified_skills_05_12",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Pi0.5 xArm side-grasp pickplace + flywheel-acquired pour (collected
    # 2026-05-10): side-grasp pickplace v5 base (50 demos, 5 primitives) +
    # 20 flywheel-collected pour trials (40 episodes: tilt-forward to pour
    # + tilt-back upright, label-normalized to 2 canonical labels). Tests
    # whether the policy absorbs the new pour primitives without losing the
    # pickplace primitives.
    TrainConfig(
        name="xarm_pickplace_pour_05_10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Same as xarm_pickplace_to_twist_v4 but with rpy unwrap applied to the
    # twist data before merging (filter_normalize_twist now np.unwraps rx/ry/rz
    # so 180° rotations crossing 2π read as continuous sequences instead of
    # wrapping to 0 mid-rotation). Trains cleanly where v4 hit elevated loss.
    TrainConfig(
        name="xarm_pickplace_to_twist_v4_unwrap",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    TrainConfig(
        name="xarm_pick_up_top_30_primitives_v2_ah30",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotXarmPrimDataConfig(
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

    # Pi0.5 red/blue only - isolated test for steerability
    # Only "move gripper to red/blue lego block" primitives, trimmed settling zeros
    TrainConfig(
        name="pi05_red_blue_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_red_blue_only",
            assets=AssetsConfig(asset_id="maggiewang/lego_red_blue_only"),
            base_config=DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # Pi0.5 red/blue only v6 - filtered from v6_trimmed for fair comparison
    # Same preprocessing (flip, downsample, trim) as pi05_lego_primitives_v6
    TrainConfig(
        name="pi05_red_blue_v6",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="maggiewang/lego_red_blue_v6",
            assets=AssetsConfig(asset_id="maggiewang/lego_red_blue_v6"),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=128,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    #
    # RoboArena configs — upstream openpi DROID baselines. Not used by InSight
    # paper experiments. Disabled because the configs reference an older
    # RLDSDroidDataConfig API; uncomment + sync the API to re-enable.
    #
    # *roboarena_config.get_roboarena_configs(),
]

