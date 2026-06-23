# Checkpoint setup

Checkpoints &mdash; pi-0.5 base weights, fine-tuned VLAs, and the paper-experiment
results &mdash; are **not** bundled with this code release. This document covers
where to put them and how to point training/serving scripts at them.

The `checkpoints/` directory at the repo root is gitignored.

## Expected layout

```
insight/
└── checkpoints/
    └── <config_name>/
        └── <exp_name>/
            └── <step>/        # orbax (JAX) checkpoint directory
                ├── assets/         # tokenizer + norm-stats artifacts
                ├── params/         # model weights
                └── train_state/    # optimizer state (for resume; not needed for inference)
```

The config name must match one of the registered names in
`src/openpi/training/sim_configs.py` or
`src/openpi/training/xarm_configs.py`. `<exp_name>` is whatever you pass to
`--exp-name` when launching training; `<step>` is the training step at which
the checkpoint was written.

Examples:

```
checkpoints/xarm_pick_from_top_v5/my_run/15000/
checkpoints/pi05_lego_oracle_flip_140_primitives/my_run/30000/
checkpoints/xarm_pickplace_to_twist_v5_unwrap/twist_post_vla/20000/
```

## Canonical sources

Paper checkpoints live on the Stanford cluster at:

```
/viscam/projects/insight/checkpoints/<exp>/<run>/<step>/
```

A HuggingFace Hub mirror with the headline checkpoints (steerable VLAs after
each acquisition pass) will be linked here when finalized. For now: contact
the authors.

## Loading a checkpoint into the policy server

```bash
uv run python training/serve_policy.py policy:checkpoint \
  --policy.config xarm_pick_from_top_v5 \
  --policy.dir checkpoints/xarm_pick_from_top_v5/my_run/15000
```

> The `policy:checkpoint` subcommand must come **before** any `--policy.*`
> flags. This is a tyro requirement.

The server binds to `0.0.0.0:8000` (reachable from any interface). Override
the port with `--port` if you want to serve multiple policies on the same host.

## Mapping cluster scripts to checkpoint names

`cluster/sim/` and `cluster/real/` contain the SLURM submission scripts that
produced the paper checkpoints. Each script's `train.py` invocation reveals
the config name and exp-name; the resulting checkpoint path follows the
pattern above. See [`cluster/README.md`](../cluster/README.md) (when present)
for the full mapping table.

## Paper-checkpoint name map (selected)

The full list lives in the two config files; this is just a quick lookup of
the names used in the paper experiments.

| Paper experiment              | Config name                                                  | Source script                                              |
| ----------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Sim flip (base pickplace)     | `pi05_lego_human_pickplace_flip`                             | `cluster/sim/train_pi05_human_pickplace_flip.sh`           |
| Sim flip (post-flywheel)      | `pi05_lego_oracle_flip_140_primitives`                       | `cluster/sim/train_pi05_oracle_flip_primitives.sh`         |
| Sim drawer (base + close)     | `pi05_drawer_open_push_combined_04_12`                       | `cluster/sim/train_pi05_drawer_open_push.sh`               |
| Real top pickplace (base)     | `xarm_pick_from_top_v5`                                      | `cluster/real/train_xarm_pick_from_top_v5.sh`          |
| Real twist (post-VLA)         | `xarm_pickplace_to_twist_v5_unwrap`                          | `cluster/real/train_xarm_pickplace_to_twist_v5_unwrap.sh` |
| Real pour (post-VLA)          | `xarm_pickplace_pour_05_10`                                  | `cluster/real/train_xarm_pickplace_pour_05_10.sh`      |
| Real unified twist+pour       | `xarm_unified_skills_05_12`                                  | `cluster/real/train_xarm_unified_skills_05_12.sh`      |
| Real sweep                    | `xarm_scoop_to_sweep_50_04_28`                               | `cluster/real/train_scoop_to_sweep.sh`                     |

If the exact config name doesn't appear above, search both `sim_configs.py`
and `xarm_configs.py`:

```bash
grep -n 'name="xarm_' src/openpi/training/xarm_configs.py
grep -n 'name="pi05_' src/openpi/training/sim_configs.py
```

## Notes on the openpi base checkpoint

InSight LoRA-fine-tunes the public `pi_0.5` checkpoint as a starting point.
The weights are hosted on Google Cloud Storage at
`gs://openpi-assets/checkpoints/pi05_base/params` (see the `weight_loader`
field of every TrainConfig). `openpi.shared.download.maybe_download`
fetches them automatically the first time a training run starts, caching
under `$OPENPI_DATA_HOME` (defaults to `~/.cache/openpi/`).
