# 01. Project overview

[Index](README.md) · Next: [02 multi-agent systems](02-multi-agent-systems.md)

## What it is

**Multi-Agent Warehouse** is a small multi-robot warehouse on an **8 x 8** grid. Several robots
(default 3, up to 6) each have one target cell. A robot that ends a step on its target scores a
**delivery** and immediately gets a new target. Robots cannot share a cell, so they must avoid each
other and the shelves.

It exists to show two things:

1. a **correct PettingZoo `ParallelEnv`** (see [06](06-pettingzoo-and-gymnasium.md)), and
2. a **clean Pygame renderer and demo** (see [10](10-rendering-pygame.md)),

in about 450 lines of library code you can read in one sitting. A stretch stage trains a PPO agent
(see [04](04-ppo.md) and [07](07-parameter-sharing-and-supersuit.md)).

The map lives in `multi_agent_warehouse/config.py` and is **fixed** (one layout, see
[14 generalization](14-generalization.md) for what that means for trained weights):

```
........        .  free floor
.##..##.        #  shelf (blocks movement)
.##..##.
........
........
.##..##.
.##..##.
........
```

Four 2 x 2 shelf blocks with aisles between them, so robots have to meet and negotiate. 48 of the 64
cells are free floor *(verified)*.

## How the pieces fit

```
                    +--------------------------------------------+
                    |          MultiAgentWarehouseEnv            |
                    |    (env.py, a PettingZoo ParallelEnv)      |
                    |                                            |
 actions dict ----> |  validate -> resolve_moves -> targets ->   | ---> obs, rewards,
 {robot_0: 4, ...}  |  rewards -> clock -> observations          |      terminations,
                    |        |                                   |      truncations, infos
                    |        +--> rules.py   (pure physics)      |      (five dicts)
                    |        +--> config.py  (map + numbers)     |
                    |        +--> render.py  (Pygame, planned)   |
                    +--------------------------------------------+
                          ^                            |
                          | act(observations)          | observations
                          |                            v
                    +--------------------------------------------+
                    |  Policy: RandomPolicy / GreedyPolicy /     |
                    |  RLPolicy   (agents.py, Stage 5)  |
                    +--------------------------------------------+

   evaluate.py             run each policy over seeded episodes, print a table
   play.py                 live window, swap policies with a key
   train.py     (planned)  train PPO with parameter sharing
```

The key idea: **everything plugs into one loop.** The demo, the evaluation and the RL agent all do
the same thing: `obs -> policy.act(obs) -> env.step(actions) -> obs ...`. Because every policy has the
same two-method interface, swapping a random policy for a trained neural network changes one line.

## File map

| File | Role | Needs Pygame? | State |
| --- | --- | --- | --- |
| `multi_agent_warehouse/config.py` | The map and every tunable number, plain constants | No | Implemented |
| `multi_agent_warehouse/rules.py` | `resolve_moves`: the physics. Pure function, no env state | No | Implemented |
| `multi_agent_warehouse/env.py` | `MultiAgentWarehouseEnv`: the PettingZoo environment | Only when rendering | Implemented (`render`, `close` are stubs) |
| `multi_agent_warehouse/render.py` | `Renderer`: draws the grid to a pixel array or window | Yes (the only module that imports it) | Planned |
| `multi_agent_warehouse/agents.py` | `RandomPolicy`, `GreedyPolicy`, `RLPolicy` | No | Planned |
| `multi_agent_warehouse/evaluate.py` | `python -m multi_agent_warehouse.evaluate` results table | No | Planned |
| `multi_agent_warehouse/train.py` | `python -m multi_agent_warehouse.train` (stretch) | No | Planned |
| `play.py` (repo root) | Interactive demo | Yes | Planned |
| `tests/test_*.py` | Tests, one file per module | Headless via `SDL_VIDEODRIVER=dummy` | `test_config`, `test_rules`, `test_env` exist |

### Why `config.py` exists

All settings (the layout, robot counts, step limit, reward values, render fps) live in one module.
`env.py` imports them and defines none, so changing the map or a reward is a one-file edit and the
grid size is **read from `LAYOUT`**, never assumed. `tests/test_config.py` guards the map: 8 x 8,
only `.` and `#`, enough free cells, and every free cell reachable from every other (a breadth-first
search), so an edited map cannot seal off a target. See [08](08-warehouse-rules.md).

## Where each source of truth lives

| Document | Path | What it holds |
| --- | --- | --- |
| Design spec | `docs/superpowers/specs/2026-10-28-multi-agent-warehouse-design.md` | Rules, observation, rewards, rendering numbers |
| Implementation plan | `docs/superpowers/plans/2026-10-28-multi-agent-warehouse.md` | 8 tasks in 5 stages and the tests each must pass |
| Design decisions | `docs/design-decisions.md` | Every decision with reasons and trade-offs |
| Packaging | `pyproject.toml` | Dependencies, the `rl` extra, pytest and ruff config |

## Implementation status

Update this table as the code lands.

| Stage | Task | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | 1 | `rules.py` + `tests/test_rules.py` (11 tests) | **Done** |
| 1 | 2 | `config.py`, `env.py` + `tests/test_config.py`, `tests/test_env.py` (4 + 19 tests) | **Done** |
| 2 | 3 | `render.py` + `tests/test_render.py` | **Done** |
| 3 | 4 | `agents.py`, `evaluate.py` + `tests/test_agents.py` | **Done** |
| 3 | 5 | `play.py` + `tests/test_play.py` | **Done** |
| 4 | 6 | README, GIF, lint and CI-equivalent checks | **Done** (README at the repo root) |
| 5 | 7 | `train.py`, `rl` extra pins, SuperSuit spike | **Done** (branch `feat/ppo-training-script`; `tests/test_rl.py`) |
| 5 | 8 | Trained model, `RLPolicy`, final report | **Done** (branch `feat/ppo-trained-agent`; `tests/test_rl.py`) |

Stage 4 is the first **releasable stopping point** (the scripted version is complete); Stage 5 is a
stretch goal.

## Check yourself

1. Which module may import Pygame, and why only one?
2. Why does `env.py` define no numbers of its own?
3. What do the demo, the evaluation and the RL agent have in common?
