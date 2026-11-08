# 05. Reward shaping

[Index](README.md) · Previous: [04 PPO](04-ppo.md) · Next: [06 PettingZoo and Gymnasium](06-pettingzoo-and-gymnasium.md)

## The problem: sparse reward

The delivery reward (+1) is **sparse**. On an 8 x 8 map with 3 robots, a robot moving at random may
take a very long time to reach its target by accident. Until it does, every step looks the same
(`-0.01`), so there is almost nothing for a learning algorithm to latch onto.

**Shaping** adds a small hint on every step so progress is rewarded even before the first delivery.

## What the project does

When `MultiAgentWarehouseEnv(shaping=True)`, each step adds

```
shaping = SHAPING_SCALE x (distance to target before the step - distance to target after the step)
```

with `SHAPING_SCALE = 0.02` and **Manhattan distance** (`|row difference| + |column difference|`).
From `env.py` *(verified)*:

```python
if self.shaping:
    before = _manhattan(old_positions[agent], self.targets[agent])
    after = _manhattan(new_positions[agent], self.targets[agent])
    reward += SHAPING_SCALE * (before - after)
```

| Move | Shaping |
| --- | --- |
| One cell closer to the target | +0.02 |
| One cell farther | -0.02 |
| Stay, bump a wall, or get blocked | 0 |

`before` and `after` are both measured against **the target held at the start of the step**, so the
delivery step still gets the final +0.02 (and not a jump caused by the new target).

**Rules:** shaping is **off by default**, **on during training only**, and **always off during
evaluation**, so reported scores measure the real task.

## Why "potential-based" matters

Careless shaping can teach the wrong thing. Imagine rewarding "being near the target": a robot might
learn to hover next to it forever and never deliver. That is **reward hacking**.

Ng, Harada and Russell (1999) showed a safe form. If the extra reward is

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

for some function `Phi` of the state (a **potential**), the optimal policy does not change.
Intuition: along any path the bonuses **telescope**. Adding up `Phi(s1) - Phi(s0)`,
`Phi(s2) - Phi(s1)`, ... leaves only `Phi(end) - Phi(start)`, which does not depend on the route.
So no route is rewarded for dawdling.

Here `Phi(s) = -0.02 x distance(s)`, and with `gamma = 1` the bonus is `0.02 x (before - after)`,
exactly what the project does. So the project's shaping is potential-based.

## Honest caveats

1. **The theorem is exact only for `gamma = 1`.** PPO's default is `gamma = 0.99`, so the shaping
   is only approximately policy-neutral. The error is small here.
2. **The target changes after each delivery, so the potential "jumps".** Summed over one target's
   life the shaping adds `0.02 x (starting distance to that target)`. On the 8 x 8 map the largest
   possible Manhattan distance is 14 *(verified)*, so that is at most `0.02 x 14 = 0.28` extra per
   delivery, on top of the +1. It is a distance-proportional bonus, not a pure relabelling. The spec
   says shaping "should not change which policy is best"; that is exact *within* one target and only
   approximately true across deliveries. Robots cannot choose their targets, so the practical effect
   should be small. It is worth knowing before you quote the theorem.
3. **Manhattan distance is only a good potential when shelves do not force detours.** The next
   section explains why this is fine now and a problem later.

## Why Manhattan works on this map but would not everywhere

I compared Manhattan distance with true shortest-path distance (a breadth-first search through the
free cells) over every pair of free cells on the current layout *(computed)*:

- 48 free cells, 2,256 ordered pairs.
- The shortest path is **never more than 2 steps longer** than the Manhattan distance.
- Only **7.1%** of pairs need any detour at all.

The open aisles around the shelf blocks mean "closer in a straight line" usually means
"closer by path". Manhattan shaping therefore points the right way almost everywhere.

On a layout with a dead end it does not. Consider this small map, robot at `(2,1)`, target at `(2,3)`,
with a wall at `(2,2)` between them *(computed)*:

```
#####
#...#       row 1:  free (1,1) (1,2) (1,3)
#.#.#       row 2:  robot (2,1), wall (2,2), target (2,3)
#.#.#
#...#       row 4:  free
```

- Manhattan distance to the target: 2. True path length (up and over, or down and under): **4**.
- Stepping **up** to `(1,1)`: Manhattan goes 2 to 3, so shaping gives **-0.02**, but the true path
  length goes 4 to 3, the *correct* move.

Manhattan shaping would **penalise the right move**. That is why [14 generalization](14-generalization.md)
recommends switching the potential to the true path distance if the layout ever changes.

## How big should the coefficient be?

Compare the per-step numbers (all from `config.py`):

- Step penalty `-0.01`, shaping `+0.02` when approaching: a robot walking straight at its target
  nets `+0.01` per step, positive but gentle.
- Moving away costs `-0.02 - 0.01 = -0.03` per step.
- A full 14-cell walk earns `+0.28` of shaping before the `+1`.

If the shaping were much larger, "approach" would drown out "deliver" and the robot might prefer
hovering. If much smaller, it would not help. The plan keeps `SHAPING_SCALE` adjustable (Task 8)
in case training needs tuning.

## Check yourself

1. Why is a robot rewarded for hovering next to its target a sign of *bad* shaping?
2. What does "telescoping" mean, and why does it make potential-based shaping safe?
3. Why is the project's shaping only approximately policy-neutral?
4. On the 8 x 8 map, why is Manhattan distance almost as good as true path distance?
5. In the dead-end example, why does Manhattan shaping punish the correct move?
6. Why must evaluation run with shaping off?
