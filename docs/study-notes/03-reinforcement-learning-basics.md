# 03. Reinforcement learning basics

[Index](README.md) · Previous: [02 multi-agent systems](02-multi-agent-systems.md) · Next: [04 PPO](04-ppo.md)

## The loop

```
        +-----------+   action a_t    +-------------+
        |   Agent   | --------------> | Environment |
        |  (policy) | <-------------- |             |
        +-----------+  obs o_{t+1},   +-------------+
                       reward r_{t+1}
```

At every step *t* the agent sees an observation, picks an action, and the environment returns the
next observation and a reward. An **episode** is one run from `reset()` to the end. In this project
an episode is 200 steps (`DEFAULT_MAX_STEPS` in `config.py`).

## Vocabulary

| Term | Meaning | In this project |
| --- | --- | --- |
| **Policy** pi(a given o) | The rule mapping observation to action (or to probabilities over actions) | `RandomPolicy`, `GreedyPolicy`, `RLPolicy` |
| **Reward** r | Immediate scalar feedback | +1 delivery, -0.01 per step, -0.1 collision |
| **Return** G | Sum of future rewards, usually discounted | Roughly: deliveries minus penalties over 200 steps |
| **Discount** gamma | How much future reward is worth now (0 to 1) | SB3's PPO default is 0.99 (not overridden by the plan) |
| **Value function** V(o) | Expected return from an observation if you keep following the policy | Learned by PPO's *critic* ([04](04-ppo.md)) |
| **Action value** Q(o, a) | Expected return after taking action `a` first | Not used directly by PPO |
| **Advantage** A(o, a) | `Q(o, a) - V(o)`: was this action better than average? | The signal PPO learns from |
| **Exploration** | Trying actions to discover what works | PPO samples actions stochastically |
| **Episode / horizon** | One run / its length | `max_steps = 200` |

## The goal

Find a policy that maximises expected return. Here: **deliver as many items as possible in 200 steps
with as few collisions as possible.**

## A worked return

Suppose a robot needs 10 steps to reach its target, and nothing else happens.

- Steps 1 to 9: each gives `-0.01`.
- Step 10: `+1.0 - 0.01 = +0.99`.
- **Undiscounted return**: `9 x (-0.01) + 0.99 = 0.90`.
- **Discounted return** with `gamma = 0.99` *(computed)*: about `0.82`. The late reward is worth a
  little less, and the early penalties are worth a little more.

Discounting matters because `0.99^100 = 0.37`: a reward 100 steps away counts for about a third.
With `gamma = 1` the agent would value a delivery 190 steps away as much as one right now.

## Terminated vs truncated

Gymnasium and PettingZoo distinguish two ways an episode ends:

- **Terminated**: the task itself ended (goal reached, agent died).
- **Truncated**: an external limit cut it off (a time limit).

This project has **no terminations**. A warehouse never "finishes", so every robot is *truncated*
together when `step_count` reaches `max_steps`. The distinction matters to learning algorithms: at a
truncation the future was cut off artificially, so a learner should still **bootstrap** a value for
the last state (estimate "what would I have earned afterwards?") rather than treat it as a true end
where the future return is zero. It also helps SuperSuit, which needs every agent to end at the same
moment ([07](07-parameter-sharing-and-supersuit.md)).

In the real env (`env.py`) the last step returns `terminations` all `False`, `truncations` all
`True`, and sets `self.agents = []` *(verified)*.

## Two families of algorithm

- **Value-based** (for example DQN): learn a value for each action, then act greedily.
- **Policy-gradient** (for example PPO): learn the policy directly by nudging the probability of
  actions that led to above-average return.

The project uses PPO. See [04 PPO](04-ppo.md).

## Exploration vs exploitation

An agent that only repeats what it already thinks is best never discovers anything better; one that
only explores never cashes in. PPO explores by **sampling** from its action probabilities, so a
robot that is 70% sure "right" is best still tries other moves 30% of the time. This is also why
`RLPolicy` samples at demo time ([11](11-policies-and-demo.md)).

## Sparse rewards

A +1 that arrives only on delivery is **sparse**: a robot wandering at random on an 8 x 8 grid may
go a long time before its first delivery, so there is little to learn from. The usual remedy is
reward shaping, covered in [05](05-reward-shaping.md).

## Check yourself

1. What is the difference between a policy and a value function?
2. Why is a time limit a truncation and not a termination?
3. What would `gamma = 0` make the agent care about?
4. Why does the project use sampling (exploration) even when demonstrating a trained policy?
5. Compute the undiscounted return of a robot that collides once (`-0.1`) and delivers on step 10.
