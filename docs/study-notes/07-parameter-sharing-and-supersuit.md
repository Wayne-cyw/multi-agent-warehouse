# 07. Parameter sharing and SuperSuit

[Index](README.md) · Previous: [06 PettingZoo and Gymnasium](06-pettingzoo-and-gymnasium.md) · Next: [08 warehouse rules](08-warehouse-rules.md)

> **Status:** verified on 2026-11-07 (Task 7). With SuperSuit 3.11.0, Stable-Baselines3 2.9.2 and
> PyTorch 2.15.0 alongside PettingZoo 1.27.0 and Gymnasium 1.4.0, the two calls below work and PPO trains.
> One catch: `PPO(..., seed=...)` fails because SuperSuit's `ConcatVecEnv` has no `seed()`; `train.py`
> seeds with SB3's `set_random_seed` and `vec_env.venv.reset(seed=...)` instead (D-42).

## The idea

You have 3 identical robots and want one good robot behaviour. **Parameter sharing** means all
robots use **one set of network weights**:

```
robot_0 obs ----\
robot_1 obs -----+--->  one shared network  --->  an action for each robot
robot_2 obs ----/
```

- Each robot feeds in its **own** observation and gets its **own** action.
- Every robot's experience is used to train the **same** weights.

Why it fits this project: the robots are interchangeable (same observation format, same actions,
same goal), so there is one skill to learn and 3 robots' worth of experience per step to learn it from.

## The practical problem

SB3 only understands **single-agent** environments. A PettingZoo `ParallelEnv` has several agents.
SuperSuit bridges the gap by pretending each robot is its own little single-agent environment.

## The two SuperSuit calls (from the plan)

```python
vec = supersuit.pettingzoo_env_to_vec_env_v1(env)
vec_env = supersuit.concat_vec_envs_v1(vec, n_envs, num_cpus=1, base_class="stable_baselines3")
model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
```

1. `pettingzoo_env_to_vec_env_v1`: turns the multi-agent env into a **vectorised** env in which
   **each robot is one sub-environment**.
2. `concat_vec_envs_v1(vec, n_envs, num_cpus=1, ...)`: makes `n_envs` copies and presents them in the
   form SB3 expects (`num_cpus=1` means the copies run in one process).
3. `PPO` then sees `3 robots x n_envs copies` independent single-agent environments, all feeding
   **one network**.

```
   copy 1:  robot_0  robot_1  robot_2  --+
   copy 2:  robot_0  robot_1  robot_2  --+--->  24 "sub-environments" (n_envs = 8)  ---> ONE PPO network
   ...                                     |
   copy 8:  robot_0  robot_1  robot_2  --+
```

This count is what drives PPO's rollout size, and why the plan's 300,000 steps are only about seven
policy updates with default settings ([04, section 8](04-ppo.md#a-budget-problem-the-plan-does-not-yet-mention-computed)).

## What this demands of the environment

| Requirement | How this project meets it |
| --- | --- |
| Every agent has the **same** observation and action space | All robots use the same `Box(-1, 1, (8,))` and `Discrete(5)` |
| The set of live agents is **fixed** during an episode (no agent dies early) | There are **no terminations**; every robot is truncated at the same step ([03](03-reinforcement-learning-basics.md)) |
| Observations are plain arrays | 8 `float32` values |
| The observation **shape** never changes | True today; also a constraint if the layout becomes variable ([14](14-generalization.md)) |

## At run time: `RLPolicy`

When the trained model plays (`RLPolicy`), the shared network is called **once per step on a
batch**:

1. Stack all robots' observations into one array (sorted by robot id, so the order is stable).
2. `model.predict(batch, deterministic=False)`: one call, one action per row.
3. Map the rows back to robot names.

`deterministic=False` samples from the action probabilities. With shared weights and similar
observations, always picking the most likely action can make two robots mirror each other into a
standoff; sampling breaks the symmetry ([11](11-policies-and-demo.md)).

## Limits of parameter sharing

- **No specialisation**: every robot behaves the same way given the same observation. Fine for
  identical robots, wrong if roles should differ.
- **Symmetry**: identical inputs give identical outputs (see above).
- **Non-stationarity**: each robot's environment includes the others, who change as the shared
  policy changes. PPO assumes a fixed environment, so this is a mild violation; in practice shared-weight
  PPO has worked well in cooperative tasks ([17](17-further-reading.md)).

## The risk, and the plan's response

SuperSuit, SB3, PettingZoo and Gymnasium versions must agree, and that is the biggest unknown in the
project. The plan makes the **first** RL task a *spike*: install the `rl` extra and run 2,048 training
steps. If that fails and the fix is not a small version change, the plan says **stop and ask**; the
fallback is a hand-written PyTorch PPO, which is a separate piece of work (design decision D-42 in
[`../design-decisions.md`](../design-decisions.md)).

## Alternatives you will meet in the literature

| Approach | Idea |
| --- | --- |
| **Independent learners** | A separate network per agent |
| **CTDE** (MADDPG, MAPPO, QMIX) | Train with extra global information, act from local observations |
| **Centralised policy** | One policy over the joint action (exponential growth: 5^3 = 125 for 3 robots) |

## Check yourself

1. What does SuperSuit turn each robot into?
2. With 3 robots and `n_envs = 8`, how many sub-environments does PPO see?
3. Why do all robots need identical observation and action spaces?
4. Why does "no terminations" help here?
5. Why is `RLPolicy` called with one batched `predict` call rather than once per robot?
6. What is the project's response if the dependency spike fails?
