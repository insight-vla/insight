# Cluster launch scripts

This directory holds the SLURM submission scripts that produced the paper
checkpoints. They are kept for reference, not as ready-to-run launchers
&mdash; every script assumes a specific cluster layout (Stanford SC compute,
`viscam` partition) and the legacy directory layout (`scripts/train.py`)
that this release reorganises into `training/`. Set `WANDB_API_KEY` and
`HF_TOKEN` in your environment before launching.

To re-run a script in this layout, change the `cd` target and the
`scripts/` paths to `training/`. Example:

```bash
# Old (in the .sh files as shipped):
cd ${USER_SCRATCH:-$HOME}/Workspace/openpi
uv run scripts/train.py pi05_lego_oracle_flip_140_primitives ...

# New equivalent under insight:
cd <repo>/insight
uv run python training/train.py pi05_lego_oracle_flip_140_primitives ...
```

## Layout

```
cluster/
├── sim/    # LIBERO (Franka pi-0.5)
└── real/   # xArm pi-0.5
```

## Paper-checkpoint &harr; script mapping

### Simulation (LIBERO)

| Paper experiment / checkpoint                | Script                                                  |
| -------------------------------------------- | ------------------------------------------------------- |
| Sim pickplace base (Fig. 3 starting point)   | `sim/train_pi05_human_pickplace_flip.sh`                |
| Sim flip post-flywheel (140 primitives)      | `sim/train_pi05_oracle_flip_primitives.sh`              |
| Sim drawer open + push (drawer-close)        | `sim/train_pi05_drawer_open_push.sh`                    |
| LIBERO base LoRA                             | `sim/train_pi05_libero_lora.sh`                         |
| Tilted pickplace (primitive ablation)        | `sim/train_pi05_pickplace_tilted_primitives.sh`         |
| Primitive benchmark                          | `sim/benchmark_primitives.sh`                           |

### Real (xArm)

| Paper experiment                   | Script                                                  |
| ---------------------------------- | ------------------------------------------------------- |
| Pick from side v5 (base)           | `real/train_xarm_pick_from_side_v5.sh`                  |
| Pick from top v5 (base)            | `real/train_xarm_pick_from_top_v5.sh`                   |
| Pickplace &rarr; twist (post-VLA)  | `real/train_xarm_pickplace_to_twist_v5_unwrap.sh`       |
| Pickplace &rarr; pour (post-VLA)   | `real/train_xarm_pickplace_pour_05_10.sh`               |
| Unified twist+pour                 | `real/train_xarm_unified_skills_05_12.sh`               |
| Scoop (base, 100 primitives)       | `real/train_xarm_scoop_100_primitives.sh`               |
| Sweep ablation (100 primitives)    | `real/train_xarm_sweep_100.sh`                          |
| Sweep (scoop &rarr; sweep retrain) | `real/train_scoop_to_sweep.sh`                          |

## Caveats

- Scripts use `cd ${USER_SCRATCH:-$HOME}/Workspace/openpi` or `xarm-openpi`.
  Edit to `cd <path-to-insight>` and change `scripts/` to `training/`.
- `assets/<config>/<dataset>/norm_stats.json` is computed lazily &mdash; the
  scripts check for its presence before kicking off training.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is a JAX memory-fraction hint; lower
  it if you see OOMs on smaller GPUs.
- Wall-clock per checkpoint varies from a few hours (LoRA on 1xL40S) to a
  full 48h limit (large primitive datasets on multi-GPU FSDP).
