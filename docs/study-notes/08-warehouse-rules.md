# 08. The warehouse rules

[Index](README.md) · Previous: [07 parameter sharing](07-parameter-sharing-and-supersuit.md) · Next: [09 move resolution](09-move-resolution.md)

This note describes `multi_agent_warehouse/config.py` and `multi_agent_warehouse/env.py`
*(verified against the code)*. The move-resolution rules have their own note: [09](09-move-resolution.md).

## Grid, robots, targets

- The map is the 8 x 8 `LAYOUT` in `config.py` ([01](01-project-overview.md) shows it). `.` is free
  floor, `#` is a shelf. 48 cells are free *(verified)*.
- Coordinates are `(row, col)`, row 0 at the top.
- `n_robots` is 1 to `MAX_ROBOTS = 6` (default `DEFAULT_N_ROBOTS = 3`), named
  `robot_0 ... robot_{n-1}`.
- Robots start on **distinct random free cells**.
- Each robot has exactly one target: a random free cell **different from its own cell**. Different
  robots may share a target cell.
- When a robot delivers it immediately gets a new random free target (again different from where it
  stands).

The grid size is **read from `LAYOUT`** (`self.height, self.width = self.shelves.shape`); nothing in
`env.py` assumes 15.

## Settings in `config.py`

| Constant | Value | Meaning |
| --- | --- | --- |
| `DEFAULT_N_ROBOTS` | 3 | Default number of robots |
| `MAX_ROBOTS` | 6 | Upper limit (also the number of render colours; this count is not itself tested) |
| `DEFAULT_MAX_STEPS` | 200 | Episode length |
| `DELIVERY_REWARD` | 1.0 | Reaching the target |
| `STEP_PENALTY` | -0.01 | Every step, every robot |
| `COLLISION_PENALTY` | -0.1 | A collided robot |
| `SHAPING_SCALE` | 0.02 | Optional progress hint ([05](05-reward-shaping.md)) |
| `RENDER_MODES` | `("human", "rgb_array")` | Valid render modes |
| `RENDER_FPS` | 10 | Window frame rate |

## Actions

`Discrete(5)`. `ACTION_DELTAS` in `rules.py` is indexed by the action number:

| Action | Constant | Meaning | Delta `(drow, dcol)` |
| --- | --- | --- | --- |
| 0 | `STAY` | stay | (0, 0) |
| 1 | `UP` | up | (-1, 0) |
| 2 | `DOWN` | down | (+1, 0) |
| 3 | `LEFT` | left | (0, -1) |
| 4 | `RIGHT` | right | (0, +1) |

An explicit `STAY` makes waiting a real choice, which matters for yielding in a corridor.

## The observation

`Box(-1, 1, shape=(8,), float32)` per robot:

| Index | Content | Range |
| --- | --- | --- |
| 0 | own row / 7 | 0 to 1 |
| 1 | own col / 7 | 0 to 1 |
| 2 | (target row - own row) / 7 | -1 to 1 |
| 3 | (target col - own col) / 7 | -1 to 1 |
| 4 | up blocked? | 0 or 1 |
| 5 | down blocked? | 0 or 1 |
| 6 | left blocked? | 0 or 1 |
| 7 | right blocked? | 0 or 1 |

The divisor is `max(H, W) - 1`, which is `7` here (stored as `self._scale`). It keeps every value in
`[-1, 1]`. "Blocked" means the neighbouring cell is **off the grid, a shelf, or occupied by another
robot**.

The real code *(verified)*:

```python
obs = np.array([row, col, t_row - row, t_col - col, *blocked], dtype=np.float32)
obs[:4] /= self._scale
```

**Worked example** (this is a test, `test_observation_values`): robot at `(0, 0)`, target
`(7, 7)`:

```
row      = 0/7         = 0.0
col      = 0/7         = 0.0
d_row    = (7-0)/7     = 1.0
d_col    = (7-0)/7     = 1.0
up       = off the top edge          -> 1.0
down     = (1,0) is free             -> 0.0
left     = off the left edge         -> 1.0
right    = (0,1) is free             -> 0.0

observation = [0, 0, 1, 1, 1, 0, 1, 0]
```

A second one *(computed from the real env)*: robot at `(3, 3)` (open floor in the middle of the map), target
`(3, 4)`: `[0.4286, 0.4286, 0.0, 0.1429, 0, 0, 0, 0]`. The target is one column away (`1/7 = 0.1429`) and
no neighbour is blocked.

### Why this observation

| Property | Why it matters |
| --- | --- |
| **Egocentric target**: the offset from me, not absolute coordinates | The network need not learn "where is the target on this map" |
| **Local**: only four neighbour flags | Deliberate partial observability: other robots show up only when adjacent |
| **Small**: 8 numbers | A tiny MLP learns from it quickly |
| **Normalised** to `[-1, 1]` | Networks train much better on small, centred inputs |
| **Absolute row/column included** | Helps a robot know where it is, but lets a network memorise *this* map. See [14](14-generalization.md) |

## One step, in order

```
1. Validate actions      (raise ValueError before touching any state)
2. Resolve moves         (resolve_moves -> new positions, collided set)
3. Check targets         (who is now standing on their target?)
4. Compute rewards       (against the targets held at the START of the step)
5. Resample targets      (for robots that delivered), update counters
6. Advance the clock     (step_count += 1; truncate at max_steps)
7. Build observations    (so a robot that just delivered already sees its NEW target)
```

Building observations last matters: the observation returned from `step` must describe the world the
robot will act in next, which includes the replacement target.

## Rewards

| Event | Reward | Receiver |
| --- | --- | --- |
| Delivery (end the step on the target) | **+1.0** | that robot |
| Every step | **-0.01** | every robot |
| Collision | **-0.1** | each collided robot |
| Shaping (off by default) | +0.02 x (dist before - dist after) | that robot |

The real code builds the reward additively *(verified)*:

```python
reward = STEP_PENALTY
if new_positions[agent] == self.targets[agent]:
    reward += DELIVERY_REWARD
if agent in collided:
    reward += COLLISION_PENALTY
if self.shaping:
    reward += SHAPING_SCALE * (before - after)
```

Worked examples (each is a test):

| Situation | Reward |
| --- | --- |
| Robot moves onto its target | `1.0 - 0.01 = 0.99` |
| Two robots both move into `(0, 1)` | each `-0.01 - 0.1 = -0.11` |
| Robot at `(0, 0)` moves up (wall bump) | `-0.01` |
| Shaping on, robot `(0,0)` to target `(0,3)`, moves right | `-0.01 + 0.02 = +0.01` |

Why these magnitudes: delivery is worth about 100 idle steps, so it clearly dominates; a collision
costs about 10 steps' worth, large enough to notice, small enough that a robot still risks one for a
delivery. The rewards are **individual** ([02](02-multi-agent-systems.md)).

## Episode, state and info

- **No terminations.** All robots are truncated together when `step_count` reaches `max_steps`;
  afterwards `env.agents == []`.
- `step()` after the episode ended returns **five empty dicts** instead of raising.
- Public attributes (read by the renderer, tests and policies): `positions`, `targets`, `shelves`
  (bool array), `step_count`, `total_deliveries`, `total_collisions`, `hud_note`.
- `infos[agent]` carries `deliveries` (that robot's running count) and `collided` (bool, this step).

## Input validation

Bad input raises `ValueError` **before any state changes**:

- constructor: `n_robots` outside 1 to 6, `max_steps < 1`, an unknown `render_mode`;
- `step`: actions whose keys are not exactly the live agents, or any action that is not an integer 0
  to 4.

A detail worth noticing in `_validate_actions` *(verified)*: it explicitly rejects booleans, because
in Python `True` is an `int` (`isinstance(True, int)` is `True`) and `step({"robot_0": True})` would
otherwise be silently treated as action 1.

## Seeding and reproducibility

`reset(seed=s)` creates **one** `np.random.default_rng(s)` that drives start cells and targets and
nothing else. `reset()` with no seed *continues* that stream. Rendering never touches it.

Why this matters: RL experiments are only comparable if they are **reproducible**. Same seed means the
same episode, which is also what lets `evaluate.py` run every policy on *identical* episodes
([12](12-evaluation.md)). The rule: no global RNG (`np.random.seed`, `random.random`); every random
thing owns its own generator.

## Check yourself

1. What does the observation look like for a robot at `(7, 0)` whose target is `(7, 7)`, with
   nothing else near it? (Answer: `[1.0, 0.0, 0.0, 1.0, 0, 1, 1, 0]`: down and left are off the grid.)
2. Why is the observation built last in `step`?
3. Why is the target given as an offset instead of absolute coordinates, and what does including the
   absolute position cost?
4. A robot bumps a shelf. What does it receive?
5. Why does `_validate_actions` reject `True`?
6. What does `reset()` with no seed do?
