# Architecture

A small library: the environment, two scripted baselines and a learned policy all sit behind two
interfaces, so the demo and the evaluation share one loop.

## Module map

```
config.py      layouts and every tunable number (the only place settings live)
rules.py       resolve_moves(): pure function, no environment state
env.py         MultiAgentWarehouseEnv: a PettingZoo ParallelEnv built on rules + config
render.py      Renderer: always draws offscreen, returns an RGB array; "human" also blits to a window
agents.py      Policy protocol (reset, act); RandomPolicy, GreedyPolicy, RLPolicy; model sidecars
evaluate.py    seeded evaluation with 95% confidence intervals
train.py       trains one shared PPO / RecurrentPPO policy (optional `rl` extra)
seeds.py       random training seeds and evaluation blocks
experiment.py  multi-seed experiments, generalisation runs, select-and-test
plots.py       learning curves, comparison figures, markdown report (optional `plot` extra)
plots_generalisation.py  layout-by-robot-count heatmaps
play.py        live demo (repo root)
scripts/record_demo.py   records demo.gif from the shipped model
```

Dependencies point one way: `config` <- `rules` <- `env` <- `render`/`agents`/`train` <-
`evaluate`/`experiment` <- `plots`. Nothing below `train.py` needs PyTorch.

## The environment

`MultiAgentWarehouseEnv` (`env.py`) is a `ParallelEnv`: every robot submits an action each step.
Settings come from `config.py`; the physics of simultaneous moves is delegated to `rules.py`.

A step runs in a fixed order:

1. validate the actions;
2. resolve moves (`resolve_moves`);
3. check targets and hand out new ones on delivery;
4. compute rewards, using the targets held at the start of the step;
5. advance the clock (episodes truncate together at 200 steps);
6. build observations last, so a robot that just delivered already sees its new target.

`resolve_moves` applies two rules (two robots may not share a cell; two robots may not swap) and
repeats until nothing changes, because a robot blocked by another robot becomes a stayer and can
block others. Bumping a wall or shelf is not a collision.

Optional constructor arguments (`observation="rich"`, `collision_penalty`, `shaping_scale`,
`layouts=(...)`) leave the defaults, and the random stream with one layout, unchanged.

## Rendering

`render.py` always draws to an offscreen Pygame surface and returns an RGB array. Window mode adds
a blit at a fixed frame rate. Pygame is imported only here, and the environment builds the renderer
lazily, so an environment without a render mode never loads it. That is what makes the renderer
testable headless (`SDL_VIDEODRIVER=dummy`).

## Policies

Every policy has `reset(seed)` and `act(observations)` and sees observations only, never the
environment's internals:

- `RandomPolicy`: a uniform action per robot.
- `GreedyPolicy`: walks toward the target along the longer axis and sidesteps at random when stuck.
- `RLPolicy`: wraps a trained model and samples from its action distribution; recurrent models keep
  their LSTM state between calls.

`available_policies()` lists what can be built; the `rl` entry appears only if the model file exists
and its libraries are installed.

### Model sidecars

A model `X.zip` has a sidecar `X.json` holding its training config. `policy_env_kwargs` and
`policy_algo` read it so `evaluate.py` and `play.py` build the matching observation mode and policy
class. A model without a sidecar is treated as a plain PPO on the basic observation.

## Training

`train.py` trains one network for all robots through SuperSuit parameter sharing: each robot becomes
one sub-environment of a vectorised env, and 8 copies are concatenated (3 robots x 8 envs = 24
sub-environments feeding one network). Details that matter:

- `TrainConfig` holds the hyper-parameters; `algo="recurrent_ppo"` selects sb3-contrib's
  `RecurrentPPO` with an LSTM.
- Reward shaping is on during training only and always off in evaluation.
- The environment sets `TimeLimit.truncated` on the last step so SB3 bootstraps the value at the
  200-step cut-off (SuperSuit merges terminated and truncated into one flag).
- Seeding goes through `set_random_seed` and `ConcatVecEnv.reset(seed=...)`, because
  `PPO(..., seed=...)` fails on SuperSuit's vector env.
- `python -m multi_agent_warehouse.train` never overwrites the shipped model; its default output is
  `models/ppo_retrained`.

## Experiments and evaluation

```
config + env + train ──> experiment.run ──> runs/experiments/NAME/
                                              manifest.json, eval_curve.csv, baselines.csv,
                                              <variant>/seed_<seed>/ (logs, snapshots)
                                                  │
                      experiment.plot ────────────┤──> plots/*.png, report.md
                      experiment.generalise ──────┤──> generalisation.csv
                      experiment.select ──────────┘──> validation pick, test-block score
```

- Training seeds are random in `[0, 1000)`; evaluation blocks are 100 consecutive seeds from
  `[10000, 2^31)`, so they never overlap a training seed. Every variant in an experiment is scored on
  the same block.
- `run` is resumable: finished runs are kept, unfinished ones are retrained, and evaluations already
  in `eval_curve.csv` are not repeated.
- `select` picks the best seed on one validation block and reports it on a disjoint test block, so
  the reported score is not inflated by the choice.
- `evaluate.py` (the README table) uses fixed seeds 1000 to 1099, 3 robots, shaping off.

## Runtime contracts

Docstring preconditions and invariants are checked at runtime by `check_contracts`, but only when
`WAREHOUSE_CONTRACTS=1` is set before the package is imported. The test suite sets it in
`tests/conftest.py`. See [`code-style.md`](code-style.md).
