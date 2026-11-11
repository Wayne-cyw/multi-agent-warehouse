# Study Notes: Multi-Agent RL, PettingZoo and the Multi-Agent Warehouse

These notes teach the basics of **multi-agent systems**, **reinforcement learning (RL)**, **PPO**
and **PettingZoo**, and every idea is tied to something in this project. There is **one topic per
file**. Together they double as the explanation and documentation of the project.

Companion file: [`../design-decisions.md`](../design-decisions.md) records *why* each choice was
made. The notes explain *what the ideas are*; that file explains *what we chose*.

## Reading order

| # | File | Topic | Read it when |
| --- | --- | --- | --- |
| 01 | [project-overview](01-project-overview.md) | What the project is, how the pieces fit, implementation status | First |
| 02 | [multi-agent-systems](02-multi-agent-systems.md) | Agents, why multi-agent is harder, reward structures | Before any RL |
| 03 | [reinforcement-learning-basics](03-reinforcement-learning-basics.md) | The RL loop, return, value, discount, terminated vs truncated | Before PPO |
| 04 | [ppo](04-ppo.md) | Policy gradients, advantage, the clipped objective, SB3 settings | Before Stage 5 |
| 05 | [reward-shaping](05-reward-shaping.md) | Sparse rewards, potential-based shaping and its limits | Before Stage 5 |
| 06 | [pettingzoo-and-gymnasium](06-pettingzoo-and-gymnasium.md) | The Parallel API and the rules the API test enforces | Before reading `env.py` |
| 07 | [parameter-sharing-and-supersuit](07-parameter-sharing-and-supersuit.md) | Training many identical robots with one network | Stage 5 |
| 08 | [warehouse-rules](08-warehouse-rules.md) | Grid, config, actions, observation, rewards, episode, seeding | Alongside `env.py` |
| 09 | [move-resolution](09-move-resolution.md) | The simultaneous-move rules in `rules.py` | Alongside `rules.py` |
| 10 | [rendering-pygame](10-rendering-pygame.md) | Offscreen-first renderer, render modes | Stage 2 |
| 11 | [policies-and-demo](11-policies-and-demo.md) | Random, greedy and RL policies; the demo loop | Stage 3 |
| 12 | [evaluation](12-evaluation.md) | Seeded evaluation and what a 20-episode mean can tell you | Stage 3 |
| 13 | [testing-and-tooling](13-testing-and-tooling.md) | TDD, test files, ruff, headless Pygame | Any time |
| 14 | [generalization](14-generalization.md) | Making trained weights work on any layout | After Stage 5 |
| 15 | [glossary](15-glossary.md) | Every term, in one table | Reference |
| 16 | [exercises](16-exercises.md) | Hands-on exercises with answers | After each stage |
| 17 | [further-reading](17-further-reading.md) | Books, papers, docs | Reference |

### Suggested paths

- **"I want to understand the code"**: 01, 06, 08, 09, then 10 to 13.
- **"I want to understand the RL part"**: 02, 03, 04, 05, 07, then 14.
- **"I want the whole story"**: in order, 01 to 14.

## Status snapshot (as of 2026-11-03)

| Stage | Deliverable | State |
| --- | --- | --- |
| 1 | `config.py`, `rules.py`, `env.py` and their tests | **Implemented.** |
| 2 | `render.py` | **Implemented.** |
| 3 | `agents.py`, `evaluate.py`, `play.py` | **Implemented.** Evaluation now runs 100 episodes with 95% confidence intervals. |
| 4 | README, GIF, CI-equivalent checks | **Implemented.** The README is at the repo root. 59 tests pass; `ruff` is clean. |
| 5 | `train.py`, trained model, `RLPolicy` | **Task 7 done** (`train.py`, `rl` extra, spike passed on 2026-11-07); Task 8 (`RLPolicy`, trained model) not started. |

Where a note describes code that does not exist yet (Stages 5 and 6), it says **(planned)**. Update the table above and the
status table in [01-project-overview](01-project-overview.md#implementation-status) as stages land.

## Conventions used in these notes

- **Verified**: checked against the code in this repo or the installed library source.
- **Illustrative sketch**: code that is *not* part of the project, written to teach an idea.
- **From memory / unverified**: a fact I could not check here (for example library defaults for
  packages that are not installed). Confirm these before relying on them.
- Coordinates are `(row, col)` with row 0 at the top.
- Each topic file ends with **Check yourself** questions.

## Versions verified in this repo's `.venv` (Python 3.11)

PettingZoo 1.27.0, Gymnasium 1.4.0, NumPy 2.4.6, Pygame 2.6.1, pytest 9.2.0, ruff 0.17.3,
Pillow 13.0.0. With the optional `rl` extra (Stage 5) also installed: Stable-Baselines3 2.9.2,
SuperSuit 3.11.0 and PyTorch 2.15.0. SB3's PPO defaults quoted in note 04 were checked against the
installed signature on 2026-11-07.
