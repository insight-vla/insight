# Dataset setup

Datasets are **not** bundled with this code release. This document describes
where datasets are expected to live, what formats are used, and how to
(re)build the flywheel-curated datasets from raw rollouts.

The `data/` directory at the repo root is gitignored.

## Expected layout

```
insight/
├── data/
│   ├── lerobot/                     # LeRobot-format datasets used by training
│   │   ├── xarm_pick_from_top_v5_primitives_trimmed/
│   │   ├── xarm_pickplace_to_twist_v5_unwrap/
│   │   ├── xarm_pickplace_pour_05_10/
│   │   ├── xarm_unified_skills_05_12/
│   │   ├── xarm_scoop_to_sweep_50_04_28/
│   │   ├── lego_pickplace_tilted_150_primitives_trimmed/
│   │   ├── lego_oracle_flip_140_primitives_trimmed/
│   │   ├── drawer_open_top_50_04_07_primitives_trimmed/
│   │   └── ...
│   │   # (exact names come from each TrainConfig's data.repo_id; the
│   │   # 'maggiewang/' or 'maggie/' HF org prefix is the user's namespace
│   │   # — substitute your own when publishing to HF.)
│   └── raw/                          # raw demo recordings (HDF5, MP4, NPZ, etc.)
│       ├── xarm_twist_pre_vla_05_24/
│       └── ...
│
└── assets/
    └── <config_name>/<dataset>/norm_stats.json   # computed by training/compute_norm_stats.py
```

Each training config in `src/openpi/training/sim_configs.py` and
`src/openpi/training/xarm_configs.py` references a HuggingFace-style dataset
repo id (e.g. `maggie/xarm_pickplace_to_twist_v5_unwrap`). With `HF_TOKEN`
set, `uv run python training/compute_norm_stats.py` will pull from the
HuggingFace Hub. Otherwise place a local copy under `data/lerobot/<repo_id>/`
and point `HF_LEROBOT_HOME` (or `LEROBOT_HOME`) at it.

## Format

InSight follows the **LeRobot dataset format** throughout. Each episode is a
parquet-backed `LeRobotDataset` shard with:

- xArm LeRobot columns: `observation/exterior_image_1_left` and
  `observation/wrist_image_left` (224x224 RGB).
- LIBERO LeRobot columns: `image` and `wrist_image` (224x224 RGB; converted
  from the raw robosuite `agentview_image` / `robot0_eye_in_hand_image` obs
  during dataset creation).
- `observation/state` — xArm: 6-D EE pose (xyz mm + rpy rad) + gripper;
  LIBERO: 8-D (3-D EE position + 3-D axis-angle + 2-D gripper qpos).
- `action` — **xArm uses absolute EE pose targets** (6) + absolute gripper
  command (1) + learned progress channel (1); **LIBERO uses EE deltas**
  (6) + absolute gripper (1) + progress (1).
- `task` (the primitive label, used as the language prompt)
- `progress` (Stage-1 scalar in [0, 1], normalized in-segment timestep)

Sim (LIBERO) episodes use the same schema with `observation/state` as an
8-D vector (3-D EE position + 3-D axis-angle + 2-D gripper qpos) matching
the Franka 7-DoF state and a different `task_index` mapping.

## Base demonstrations referenced in the paper

The paper trains on the following base demonstrations (all collected before
any flywheel acquisition):

| Experiment           | Base demos                                 | Volume        |
| -------------------- | ------------------------------------------ | ------------- |
| Sim block-flip       | LIBERO pick-and-place                      | 150 demos     |
| Sim drawer-close     | LIBERO drawer-open                         | 50 demos      |
| Real twist           | xArm top-grasp pick-and-place              | 50 demos      |
| Real pour            | xArm top-grasp pick-and-place (same as twist) | 50 demos  |
| Real twist-then-pour | uses the post-VLA twist + pour datasets    | &mdash;       |
| Real sweep           | xArm scooping demos                        | (paper Table) |

## Building the flywheel-curated datasets

Successful flywheel rollouts are split into per-primitive segments and merged
back into the training set. The scripts that do this live in
`training/preprocess/`:

```bash
# Per-primitive segmentation + label normalization
uv run python training/preprocess/preprocess_all_primitives.py \
  --source data/raw/xarm_twist_pre_vla_05_24 \
  --output data/lerobot/xarm_twist_primitives

# Twist-specific filter (drops failed rollouts, clamps outliers)
uv run python training/preprocess/filter_normalize_twist.py

# Merge into the base pickplace dataset for retraining
uv run python training/preprocess/merge_pickplace_twist.py
```

Equivalent scripts exist for pour, unified skills, and the sim flip oracle.

## Computing normalization statistics

After (re)building a dataset, compute its norm stats once before training:

```bash
uv run python training/compute_norm_stats.py \
  --config-name xarm_pickplace_to_twist_v5_unwrap
```

Results are written to `assets/<config_name>/<dataset>/norm_stats.json`. The
cluster SLURM scripts in `cluster/` short-circuit this step when the file
already exists.

## Visualizing datasets

```bash
uv run python training/visualize_lerobot_dataset.py --repo_id xarm_pickplace_to_twist_v5_unwrap
uv run python training/preprocess/visualize_trimmed_demos.py --root ~/.cache/huggingface/lerobot/maggie/xarm_twist_post_vla
uv run python training/viz_open_gripper.py            # sanity-check gripper open/close transitions
```

## Notes and caveats

- **xArm RPY wrap.** Some demos have `state.rx` / `state.rz` 2&pi; wraps that
  cause training loss spikes. The `*_unwrap` dataset variants (e.g.
  `xarm_pickplace_to_twist_v5_unwrap`) are pre-unwrapped.
- **LIBERO observation quaternions are XYZW**, not WXYZ &mdash; relevant if
  you write a new policy adapter.
- **Sim flywheel datasets are large.** Each successful rollout is a few MB;
  budget tens of GB for a full sim acquisition run.
- **HuggingFace Hub** is the canonical source for the paper datasets; the
  exact repo ids are listed in the corresponding TrainConfig entries.
