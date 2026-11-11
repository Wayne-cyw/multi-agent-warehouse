# 12. Evaluation

[Index](README.md) · Previous: [11 policies and demo](11-policies-and-demo.md) · Next: [13 testing and tooling](13-testing-and-tooling.md)

> **Status:** implemented (Stage 3, Task 4), revised 2026-11-03 to 100 episodes with 95% confidence intervals.

## What the project does

```bash
python -m multi_agent_warehouse.evaluate
```

Runs each available policy for **100 episodes** by default (seeds 1000 to 1099, 3 robots, 200 steps,
shaping **off**; `--episodes N` changes the count) and prints **mean deliveries per episode** and
**mean collisions per episode**, each with a **95% confidence half-width**, in a fixed-width table. `run_episode` resets the env and the policy with the same seed; `evaluate` builds a
fresh policy and a fresh env per episode.

It exists so **every claim in the README has a measured number behind it**.

## Why these choices

| Choice | Reason |
| --- | --- |
| **Same seeds for every policy** | Each policy plays *identical* episodes (same starts, same targets), so a difference is the policy, not the luck of the draw. This is a **paired** comparison. |
| **Shaping off** | Reported reward reflects the real task, not the training hint ([05](05-reward-shaping.md)). |
| **Seeds from 1000** | Away from the small seeds used in tests and demos, so evaluation is not accidentally tuned on them. |
| **Fresh policy and env per episode** | No state leaks between episodes. |
| **Deliveries and collisions as the two metrics** | Throughput and safety, the two things the rewards trade off. |

## What a mean over n episodes can and cannot tell you

A mean over `n` episodes has a **standard error** of about `sd / sqrt(n)`, where `sd` is how much
single episodes vary. The table reports `1.96 x` that value as a 95% confidence half-width
(a normal approximation, which is reasonable at `n = 100`). With the original `n = 20`, the mean was
only known to within roughly one fifth of a typical episode-to-episode spread; `n = 100` makes the
interval about 2.2 times narrower.

Current numbers: random delivers 2.63 ± 0.34 per episode and greedy 12.67 ± 2.07, so the gap is far
outside the noise. Collisions are 12.80 ± 1.49 and 401.30 ± 24.24.

- It **can** separate policies that are far apart: the plan's test requires greedy to deliver at least
  2x as many as random.
- It **cannot** separate two similar policies if their intervals overlap. Treat overlap as "cannot
  tell apart", not as "equal".
- It reports **no deadlock flag** (a non-goal), and the intervals treat each policy separately rather
  than testing the paired difference.

If a README claim depends on a small gap, run more episodes first (`--episodes 500`).

## The RL acceptance bar

Decided **before** looking at results: RL's mean deliveries must be at least **2x random's**. Beating
greedy is not required, and the README must say honestly where RL lands, plus what it was not tested
on (one layout, 3 robots, no generalisation, see [14](14-generalization.md)). Fixing the success
criterion in advance prevents moving the goalposts.

## Training reward is not evaluation reward

PPO's logged training reward (`ep_rew_mean`) is measured **with shaping on**, and on episodes from
whatever seeds the training run used. Never compare policies with it ([04](04-ppo.md)). Compare them
with this evaluation: shaping off, fixed seeds, identical episodes.

## Check yourself

1. Why does every policy play the same seeds?
2. Why is shaping off during evaluation?
3. What does a standard error of about `sd / sqrt(n)` imply about small differences, and what does overlapping intervals mean?
4. What is the RL acceptance bar, and why is it fixed beforehand?
5. Why is the training reward the wrong number for comparing policies?
