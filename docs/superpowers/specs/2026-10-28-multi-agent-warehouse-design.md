# Multi-Agent Warehouse: design

> **Update 2026-11-03:** Stages 1 to 4 are built and merged; Stage 5 (RL) is not started. This
> document was written before the build, and mentions of `warehouse_rl`, `mvp-build` and
> `DESIGN.md` refer to a separate project that is not part of this repository. The README now lives
> at the repo root, and evaluation was changed to 100 episodes with 95% confidence intervals
> (see the Evaluation section).

A deliberately small multi-robot warehouse that showcases two things: a correct PettingZoo
`ParallelEnv`, and a clean Pygame renderer and demo. It sits beside the existing `warehouse_rl`
package and does not touch it. Target size is about 450 lines of library code.

## Intent and the why

`warehouse_rl` (the MVP on `mvp-build`) is rigorous but large: three layouts, five baselines, a
deadlock detector, a statistics harness and 17 test files. A reviewer cannot take it in at a
glance. This version exists so that someone can clone the repo, run one command, and within 30
seconds see a working multi-agent environment and a polished window, with the code short enough
to read in one sitting. `DESIGN.md` remains the pointer to the deeper work.

## Goals

- A PettingZoo `ParallelEnv` that passes `pettingzoo.test.parallel_api_test`.
- Simple, fully tested rules: walls and shelves block, robots cannot share a cell.
- A Pygame renderer that works in a window (`human`) and offscreen (`rgb_array`), headless-safe.
- Two scripted policies (random, greedy) and a live demo (`play.py`) that switches between them.
- A tiny evaluation that prints mean deliveries and collisions per policy over seeded episodes.
- A README with a screenshot or GIF and a five-line design explanation.
- Stretch (last stage): a PPO agent with parameter sharing, trained once, weights committed,
  selectable in `play.py` and included in the evaluation table.

## Non-goals

- Deadlock detection, CI-grade evaluation (confidence intervals were added 2026-11-03), several layouts at once (the map
  is swappable by editing `config.py`, but only one is supported and tested), fixed-task mode.
- A centralized planner, a yielding baseline, any claim about generalization.
- Re-implementing PPO. The learner is Stable-Baselines3.
- Touching or refactoring `warehouse_rl/`.

## Anti-goals

- A stage that cannot be run. Every stage ends with something demonstrable.
- A demo that crashes when optional pieces (the RL model file) are missing.
- Reporting an RL result that was not measured, or claiming it beats greedy unless it does.

## Constraints

- Python 3.10+; project venv at `.venv` (Python 3.11).
- Library dependencies: the ones `pyproject.toml` already declares (NumPy, Gymnasium, PettingZoo,
  Pygame). Pillow (a `dev` extra in `pyproject.toml`; the original sentence assumed matplotlib installs it) is used for the README GIF only.
- RL dependencies go in a new optional extra `rl` (`stable-baselines3`, `supersuit`, `torch`) and are
  imported lazily so the scripted version never needs them.
- All randomness from `np.random.default_rng`; no global RNG.
- Lint and format with ruff exactly as CI does (`E,F,I,B,UP`, line length 100, `ruff format --check`).

## The game

**Grid.** One fixed 8 x 8 layout, `.` free and `#` shelf, four 2 x 2 shelf blocks:

```
........
.##..##.
.##..##.
........
........
.##..##.
.##..##.
........
```

**Settings file.** The layout and every tunable number live in one module,
`multi_agent_warehouse/config.py`: `LAYOUT`, `DEFAULT_N_ROBOTS`, `MAX_ROBOTS`, `DEFAULT_MAX_STEPS`,
the four reward values, `RENDER_MODES` and `RENDER_FPS`. The environment, policies, evaluation and
demo import from it and define no settings of their own; the grid size is read from `LAYOUT`, so
editing the map is a one-file change (tests check it is rectangular, uses only `.` and `#`, and has
no sealed-off cells). Drawing sizes (cell size, colours) stay in `render.py`.

**Robots.** `n_robots` (default 3, allowed 1 to 6) with ids `robot_0` ... `robot_{n-1}`. They start
on distinct random free cells. Each has one target: a random free cell different from its own
cell. Targets of different robots may coincide.

**Actions.** `Discrete(5)`: 0 stay, 1 up (row - 1), 2 down (row + 1), 3 left (col - 1),
4 right (col + 1). Same convention as `warehouse_rl`.

**Observation.** `Box(-1, 1, shape=(8,), float32)` per robot, built from its own state:

| Index | Content |
| --- | --- |
| 0-1 | own row, own column, each divided by `max(H, W) - 1` |
| 2-3 | target offset (target - own) in rows and columns, divided by `max(H, W) - 1` |
| 4-7 | neighbour blocked flags for up, down, left, right: 1.0 if out of bounds, shelf, or occupied by another robot, else 0.0 |

**Move resolution** (pure function `resolve_moves`, two rules):

1. *Same cell.* A robot that moves into a cell that another robot also proposes is blocked. A
   robot that stays (or bumps a wall) proposes its own cell, so moving into a stayer is the same rule.
2. *Swap.* Two robots exchanging cells are both blocked.

A blocked robot stays where it is, and then counts as a stayer, so a robot moving into its cell is
blocked too (repeat to a fixed point). Following a robot that moves away is allowed. A rotation of
four robots around a 2 x 2 block is allowed (a deliberate simplification versus `warehouse_rl`).
Bumping a shelf or the grid edge makes the robot stay and is not a collision.

**Collision.** A robot *collides* in a step if it tried to move and was blocked by another robot,
directly or through the repeat rule. Collided robots get the collision penalty. Wall bumps and
robots that chose to stay never collide.

**Step order.** Validate actions, resolve moves, check targets, compute rewards, advance the clock,
build observations. A robot that ends the step on its target delivers: it gets a new target
immediately and that robot's observation shows the new target.

**Rewards** (individual):

| Event | Reward | Receiver |
| --- | --- | --- |
| Delivery (end the step on the target) | +1.0 | that robot |
| Every step | -0.01 | every robot |
| Collision | -0.1 | each collided robot |
| Shaping (off by default) | +0.02 x (Manhattan distance to the target before the step - after the step), measured against the target held at the start of the step | that robot |

Shaping is potential-based (a function of distance only), so it should not change which policy is
best. It exists for the RL stage and is never used when evaluating.

**Episode.** No terminations. All robots are truncated together when the step count reaches
`max_steps` (default 200). Afterwards `env.agents` is empty.

**State and info.** Public attributes the renderer, tests and policies read: `positions`,
`targets` (`dict[str, tuple[int, int]]`), `shelves` (bool array), `step_count`,
`total_deliveries`, `total_collisions`. `infos[agent]` carries `deliveries` (that robot's running
count) and `collided` (bool, this step).

**Seeding.** `reset(seed=s)` builds one `np.random.default_rng(s)` that drives starts and targets
and nothing else. `reset()` without a seed continues the stream. Rendering never touches it.

## Rendering

`Renderer` always draws to an offscreen `pygame.Surface` and returns an `(H_px, W_px, 3)` uint8
array; in `human` mode it also blits to a window at `render_fps` (10). Layout: a 40 px HUD bar on
top, then the grid at 64 px per cell (a 512 x 552 px frame). Floor light grey, shelves dark brown,
each robot a filled circle in its own colour, each target an outlined square inset 8 px in the same
colour (line width 6). The HUD
shows step, deliveries, collisions and a free-text note (`env.hud_note`, set by `play.py`, for
example the policy name and "PAUSED"). `render_mode=None` makes `render()` a no-op returning
`None` and opens no window. Pygame is imported lazily, only by the renderer.

## Policies

Interface: `reset(seed)` and `act(observations) -> dict[str, int]`, using only observations.

- `RandomPolicy`: uniform action per robot from its own seeded RNG.
- `GreedyPolicy`: step along the axis with the larger absolute target offset (ties go to rows).
  If that step is blocked, try the other axis if its offset is non-zero. If still blocked, or the
  other offset is zero, take a random unblocked direction (own RNG). If none are unblocked, or the
  robot is on its target, stay. The random sidestep is what gets greedy around shelves; it is not
  privileged, because it uses only the observation.
- `RLPolicy` (stretch): loads a saved model, predicts for all robots in one batch (shared weights),
  samples actions (`deterministic=False`) so identical robots do not lock into a symmetric standoff.

## Evaluation

`python -m multi_agent_warehouse.evaluate` runs each available policy for 100 episodes by default (seeds 1000 to 1099; `--episodes N` changes it,
3 robots, 200 steps, shaping off) and prints mean deliveries per episode and mean collisions per
episode, each with a 95% confidence half-width. No deadlock flag. (The original spec said 20 episodes and no intervals.) It exists so any claim in the README has a number behind it.

## Demo

`play.py` opens the window and runs a policy. Keys: space pause, R reset, P cycle policy
(random, greedy, and rl when its model file exists), Esc or Q quit. Flags: `--policy`, `--robots`,
`--seed`, `--fps`, `--frames N` (stop after N frames, for headless smoke tests). A finished episode
auto-resets with the next seed. If the RL model file is missing, the demo prints a one-line hint
and carries on without the `rl` mode.

## RL stage (stretch)

PPO from Stable-Baselines3 with parameter sharing through SuperSuit: the parallel env is turned
into a vectorized env where each robot is one sub-env, so a single network is trained on all robots'
experience. `python -m multi_agent_warehouse.train` trains with shaping on, saves `models/ppo_multi_agent_warehouse.zip`, and that zip
is committed (expected under 1 MB) so a clone can play it without training. Acceptance bar: mean
deliveries at least 2 x random's. Beating greedy is not required, and the README says honestly
where RL lands.

**Gate:** the first RL task is a spike (install and a 2,048-step run). If the stack does not work
on this machine's package versions after a small pinning effort, stop and ask: the fallback is a
hand-written PyTorch PPO, which is a separate piece of work.

## Stages

1. Rules and environment (`rules.py`, `env.py`, tests).
2. Renderer (`render.py`, `env.render`).
3. Policies, evaluation and demo (`agents.py`, `evaluate.py`, `play.py`).
4. Ship the scripted version (README, GIF, lint and CI-equivalent checks).
5. RL (stretch): spike, then train, commit the model, wire into demo and evaluation.

## Decisions made by me (confirmed by you on 2026-11-03, except the last)

- New package `multi_agent_warehouse/` in the same repo, on a new branch `multi-agent-warehouse`.
- Observation is 8 values with neighbour flags rather than the 5 x 5 windows of `warehouse_rl`.
- Collision rule is two rules plus a fixed point, and rotations are allowed.
- Collision penalty also goes to robots held up behind a conflict (simpler than `warehouse_rl`,
  which exempts them).
- GIF written with Pillow, which matplotlib already installs; needs your OK, as it did in
  `warehouse_rl` (decision 15).
- `play.py` lives at the repo root (the entry point a reviewer runs); training and evaluation are modules (`python -m multi_agent_warehouse.train`, `python -m multi_agent_warehouse.evaluate`).
- Optional keyboard-controlled robot is out of the plan; it can be added after stage 4.
