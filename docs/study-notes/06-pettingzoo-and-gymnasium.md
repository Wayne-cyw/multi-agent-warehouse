# 06. PettingZoo and Gymnasium

[Index](README.md) · Previous: [05 reward shaping](05-reward-shaping.md) · Next: [07 parameter sharing](07-parameter-sharing-and-supersuit.md)

## Where PettingZoo sits

```
  Gymnasium          the standard API for SINGLE-agent environments
  PettingZoo         the standard API for MULTI-agent environments (built on Gymnasium spaces)
  SuperSuit          wrappers that convert PettingZoo envs into forms RL libraries accept
  Stable-Baselines3  single-agent RL algorithms (PPO, DQN, ...)
```

PettingZoo is to multi-agent RL what Gymnasium is to single-agent RL: a shared interface, so any
algorithm can train on any environment.

## Gymnasium refresher

A single-agent Gymnasium environment offers:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

and describes its inputs and outputs with **spaces**:

| Space | Meaning | Used here |
| --- | --- | --- |
| `Discrete(n)` | An integer 0 to n-1 | `Discrete(5)`: stay, up, down, left, right |
| `Box(low, high, shape, dtype)` | An array of numbers in a range | `Box(-1, 1, shape=(8,), float32)`: the observation |

## Two PettingZoo APIs

| | **AEC** (Agent Environment Cycle) | **Parallel** (this project) |
| --- | --- | --- |
| Who acts | One agent at a time, in turn | All agents at once |
| Fits | Board games, turn-based | Simultaneous-move settings (robots, particles) |
| Step | `env.step(action)` for the current agent | `env.step({agent: action, ...})` for everyone |

Our robots all move at the same instant, so a `ParallelEnv` is the natural fit. It also forces the
interesting design problem: *what happens when two robots move into the same cell?* AEC would hide
it by serialising the moves ([09](09-move-resolution.md)).

## The `ParallelEnv` contract

Everything is a **dictionary keyed by agent name**. *(Verified against PettingZoo 1.27.0's
`pettingzoo/utils/env.py`.)*

```python
env.possible_agents        # every name that can ever exist: ["robot_0", "robot_1", "robot_2"]
env.agents                 # names alive now; shrinks as agents finish; [] when the episode is over

obs, infos = env.reset(seed=None, options=None)

obs, rewards, terminations, truncations, infos = env.step(actions)
#   actions      = {"robot_0": 4, "robot_1": 0, ...}     one per live agent
#   obs          = {"robot_0": array(8,), ...}
#   rewards      = {"robot_0": -0.01, ...}
#   terminations = {"robot_0": False, ...}
#   truncations  = {"robot_0": False, ...}
#   infos        = {"robot_0": {...}, ...}

env.observation_space(agent)   # a METHOD, not an attribute
env.action_space(agent)
env.render()
env.close()
env.metadata               # {"name": ..., "render_modes": [...], "render_fps": ...}
```

A typical run:

```python
env = MultiAgentWarehouseEnv()
obs, infos = env.reset(seed=0)
while env.agents:                                    # loop until the episode is over
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

## What the API test enforces

`pettingzoo.test.parallel_api_test` (read from the installed source) checks, among other things:

- `reset` accepts `seed` and `options` arguments.
- `obs` and `infos` contain every agent in `env.agents`.
- `step` returns five dicts.
- `env.agents` is always a subset of `possible_agents`.
- A finished agent disappears from `env.agents` and does not come back within the episode.
- `observation_space(agent)` and `action_space(agent)` return **the exact same object** on every
  call (an identity check with `is`, not just equality).

The project's test runs it with `num_cycles=300`. That is more than `max_steps = 200`, so a whole
episode, including truncation, is exercised (the test also resets twice).

## Why the spaces are cached

Gymnasium spaces are seedable objects that own their own random state. If `action_space(agent)`
built a new `Discrete(5)` on every call, anything you did to a space (seeding it, sampling from it)
would be lost, and PettingZoo's identity check would fail. Caching makes the first call build the
space and every later call return the same object.

The real `env.py` *(verified)* does this with `functools.cache`:

```python
@cache  # noqa: B019 (spaces are tiny and the env lives for the program)
def observation_space(self, agent: str) -> Box:
    return Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

`functools.cache` is shorthand for `lru_cache(maxsize=None)`, the form the plan mentions.
The `noqa: B019` silences ruff's warning that caching a method keeps `self` alive: harmless for a
small object that lives as long as the program.

## `reset` and `step` in this project

`env.py` *(verified)* follows the contract:

- `reset(seed=None, options=None)`: a seed builds a fresh `np.random.default_rng(seed)`; no seed
  continues the stream. Returns `(observations, infos)`.
- `step(actions)`: if `env.agents` is empty it returns **five empty dicts** (no exception). Otherwise
  it validates, resolves moves, computes rewards and returns the five dicts. On the final step
  `truncations` are all `True` and `self.agents` is emptied.
- `options` is accepted but unused today. It is the natural place to pass a layout per episode
  later ([14](14-generalization.md)).

## `AgentID` names and why strings

Agents are named `robot_0 ... robot_{n-1}`. Strings are what the dict-keyed API expects, and they
sort predictably, which the project uses for determinism (`sorted(positions)` in `rules.py`).

## Check yourself

1. How does `env.agents` differ from `env.possible_agents` at the end of an episode?
2. What does the Parallel API return from `step`, and what are the dicts keyed by?
3. Why must `observation_space("robot_0")` return the same object on repeated calls?
4. Why does the test use `num_cycles=300` when episodes are 200 steps?
5. What does this env return from `step()` after the episode has ended?
