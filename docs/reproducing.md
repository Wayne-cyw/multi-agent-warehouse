# Reproducing the results

Every number in the README comes from a command below. Training uses random seeds on purpose, so a
fresh run will not match a published figure to the last digit; compare against the seed spread, not
a single value.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,rl,plot]"
```

Drop `rl` to skip PyTorch (scripted baselines only). `plot` is only for experiment figures.

## 1. The README results table (no training)

```bash
python -m multi_agent_warehouse.evaluate
```

Scores random, greedy and the shipped model on the same 100 seeded episodes (seeds 1000 to 1099, 3
robots, 200 steps, shaping off), reported as mean ± 95% confidence interval. Expected, from the
README:

| Policy | Deliveries / episode | Collisions / episode |
|---|---|---|
| random | 2.63 ± 0.34 | 12.80 ± 1.49 |
| greedy | 12.67 ± 2.07 | 401.30 ± 24.24 |
| shipped PPO | 61.99 ± 1.37 | 50.90 ± 4.91 |

The `rl` row needs the `rl` extra (sb3-contrib for the shipped recurrent model). The evaluation is
seeded, so it should reproduce closely on the same library versions.

## 2. Tests and lint

```bash
pytest
ruff check .
```

The suite runs headless and sets `WAREHOUSE_CONTRACTS=1` itself. RL tests skip without the `rl`
extra.

## 3. Training a model

```bash
python -m multi_agent_warehouse.train --steps 300000 --seed 0
```

Writes `models/ppo_retrained.zip` and its sidecar; it never overwrites the shipped model. Defaults
are a plain PPO with truncation bootstrapping. `--seed 0 --no-bootstrap-truncation` uses the first
version's settings, but it has not been checked to reproduce `models/ppo_v1_300k.zip` exactly.

## 4. The experiments

Each write-up in [`evidence/experiments/`](evidence/experiments/README.md)
lists its exact command, seeds and evaluation block. All share this shape:

```bash
python -m multi_agent_warehouse.experiment run --name NAME --variants V1,V2 --seeds 5 --steps 1000000
python -m multi_agent_warehouse.experiment plot --name NAME
python -m multi_agent_warehouse.experiment generalise --name NAME --variants V1,V2
python -m multi_agent_warehouse.experiment select --name NAME --variant V1
```

The shipped model's experiment (E5):

```bash
python -m multi_agent_warehouse.experiment run --name e5-tuned --variants tuned,base --seeds 5 --steps 1000000
python -m multi_agent_warehouse.experiment plot --name e5-tuned
python -m multi_agent_warehouse.experiment generalise --name e5-tuned --variants tuned,base
python -m multi_agent_warehouse.experiment select --name e5-tuned --variant tuned --committed models/ppo_v1_300k.zip
```

To rerun with the published seeds instead of random ones, pass `--train-seeds` and
`--eval-seed-base`:

| Experiment | Training seeds | Eval seed base |
|---|---|---|
| E0 (`legacy`, 300k) | 886, 523, 717, 522, 538 | 536374503 |
| E1 (`base`, 3M) | 199, 731, 848, 179, 31 | 672125400 |
| E4 (generalisation) | 715, 774, 15, 773, 203 | recorded in its `manifest.json` |
| E5 (`tuned`, `base`, 1M) | 367, 939, 330, 605, 150 | 2135743778 |

The E5 selection used validation base 1567250135 and test base 844549345
(`--validation-base`, `--test-base`). Seeds for E2 and E3 are in
[`FINDINGS.md`](evidence/results/FINDINGS.md).

Outputs go to `runs/experiments/NAME/` (git-ignored). The shipped model is the `tuned` variant,
training seed 330, 1M steps; to retrain just that run use `--seeds 1 --train-seeds 330`. Same seed
gives identical weights on the same library versions and hardware, but I have not checked that it
reproduces the committed zip byte for byte.

### Cost

The plain network trains 500k steps in about 20 seconds (E3); the LSTM variants take roughly 15 minutes per
seed (about 500 steps/s on a laptop CPU), so E5 is a few hours in total.

## 5. Regenerating the demo and figures

```bash
python scripts/record_demo.py
```

Rewrites `docs/evidence/demo.gif` from the shipped model (needs `rl` and `dev`, for
Pillow). Experiment figures come from `experiment plot`; `plots.promote` copies chosen ones into
`docs/evidence/`.
