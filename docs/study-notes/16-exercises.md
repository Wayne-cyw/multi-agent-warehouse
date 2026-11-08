# 16. Exercises

[Index](README.md) · Previous: [15 glossary](15-glossary.md) · Next: [17 further reading](17-further-reading.md)

Try each before opening the answer. The **stage** column says when you can do it; exercises needing
code that does not exist yet are marked *planned*. Answers marked *(verified)* were run against the
real code; the rest are worked by hand.

| # | Exercise | Needs |
| --- | --- | --- |
| 1 | Hand-simulate the cascade | Notes only |
| 2 | Predict the plus-shape collision | Stage 1 (done) |
| 3 | Break the swap rule on purpose | Stage 1 (done) |
| 4 | Compute an observation by hand | Stage 1 (done) |
| 5 | Reward arithmetic | Stage 1 (done) |
| 6 | GAE by hand | Notes only |
| 7 | The PPO clip, three cases | Notes only |
| 8 | PPO's rollout budget | Notes only |
| 9 | Shaping arithmetic | Notes only |
| 10 | Reward experiment | Stage 3 and 5 (planned) |
| 11 | Find a map where greedy fails | Stage 3 (planned) |
| 12 | Parameter-sharing thought experiment | Notes only |
| 13 | Add path guidance (optional) | Stage 5 done (planned) |

---

## 1. Hand-simulate the cascade

Robots `a(2,0)` RIGHT, `b(2,1)` RIGHT, `c(2,3)` LEFT on an empty row. Write the proposals, then each
round of blocking, then the final positions and who collided.

<details><summary>Answer</summary>

Proposals: `a -> (2,1)`, `b -> (2,2)`, `c -> (2,2)`. Round 1: `b` and `c` share `(2,2)`, so both are
blocked and reset to their own cells. (`a` is not blocked yet: its proposal `(2,1)` is not another
robot's proposal in the round-1 snapshot.) Round 2: `b` is now a stayer proposing `(2,1)`, the same as
`a`'s proposal, so `a` is blocked. Round 3: nothing changes. Final: nobody moved; all three collided.
*(verified against the real function)*
</details>

## 2. Predict the plus-shape collision

Four robots in a plus shape around the centre `(2,2)` of an empty 5 x 5 grid: `a(1,2)` DOWN,
`b(3,2)` UP, `c(2,1)` RIGHT, `d(2,3)` LEFT. Predict the result **before** running
`resolve_moves`, then check it.

<details><summary>Answer</summary>

All four propose `(2,2)`. The same-cell rule blocks all four in round 1, and they reset to their own
cells. Nobody moves and all four collide. *(verified)*
</details>

## 3. Break the swap rule on purpose

Copy `resolve_moves` to a scratch file and make `_is_blocked` ignore the `swap` condition. Run
`tests/test_rules.py` against it. Which tests fail, and why does "swap" fail while "follow the
leader" does not?

<details><summary>Answer</summary>

Only `test_swap` fails; the other 10 still pass *(verified by running the real test file against a
patched copy)*. With `a(2,1)` RIGHT and `b(2,2)` LEFT, the
proposals are `(2,2)` and `(2,1)`: different cells, so the same-cell rule never fires and the two
robots pass through each other. In "follow the leader" the proposals are `(2,2)` and `(2,3)`, which are
also different and are *meant* to be allowed. Lesson: the swap rule is the only thing that catches two
robots trading places.
</details>

## 4. Compute an observation by hand

Robot at `(3,3)`, target `(0,7)`, nothing else near it. What is the observation? Then compute it for a
robot at `(7,0)` whose target is `(7,7)`.

<details><summary>Answer</summary>

`(3,3)`: row `3/7 = 0.4286`, col `0.4286`, `d_row = (0-3)/7 = -0.4286`, `d_col = (7-3)/7 = 0.5714`; all four
neighbours are free floor, so the flags are `0, 0, 0, 0`:
`[0.4286, 0.4286, -0.4286, 0.5714, 0, 0, 0, 0]`.

`(7,0)`: `[1.0, 0.0, 0.0, 1.0, 0, 1, 1, 0]`: down and left are off the grid, up `(6,0)` and right
`(7,1)` are free.
</details>

## 5. Reward arithmetic

Three robots. `robot_0` reaches its target. `robot_1` and `robot_2` both move into the same free
cell. Shaping is off. What does each receive?

<details><summary>Answer</summary>

`robot_0`: `-0.01 + 1.0 = 0.99`. `robot_1` and `robot_2`: `-0.01 - 0.1 = -0.11` each.
</details>

## 6. GAE by hand

Rewards `[0, 0, 1]`, critic values `V = [0.5, 0.5, 0.5]`, `V = 0` after the end, `gamma = 1`. Compute
the advantages for `lambda = 0` and `lambda = 1`.

<details><summary>Answer</summary>

`delta = [0 + 0.5 - 0.5, 0 + 0.5 - 0.5, 1 + 0 - 0.5] = [0, 0, 0.5]`.
`lambda = 0`: `A = [0, 0, 0.5]`. `lambda = 1`: `A = [0.5, 0.5, 0.5]`, which equals `return - V`
(`1 - 0.5` at every step). With `lambda = 0` only the last step is credited; with `lambda = 1` every
step before the delivery shares the credit.
</details>

## 7. The PPO clip, three cases

With `epsilon = 0.2`, what is `min(r * A, clip(r) * A)` and is the gradient active for:
(a) `A = +1, r = 1.1`, (b) `A = +1, r = 1.3`, (c) `A = -1, r = 0.7`?

<details><summary>Answer</summary>

(a) unclipped `1.1`, clipped `1.1` (inside `[0.8, 1.2]`): value `1.1`, **active**.
(b) unclipped `1.3`, clipped `1.2`: min is `1.2`, the clipped constant, so the gradient is **zero**:
the good action is already more likely enough.
(c) unclipped `-0.7`, clipped `0.8 x -1 = -0.8`: min is `-0.8`, clipped, gradient **zero**: the bad
action is already less likely enough.
</details>

## 8. PPO's rollout budget

With `n_steps = 128`, 6 robots, 4 copies and 300,000 total steps: how many sub-environments, samples
per rollout and updates?

<details><summary>Answer</summary>

Sub-environments `6 x 4 = 24`; samples per rollout `128 x 24 = 3,072`; updates
`ceil(300,000 / 3,072) = 98`. Compare with the plan's defaults (about 7 updates,
[04](04-ppo.md)).
</details>

## 9. Shaping arithmetic

A robot walks 12 cells in a straight line to its target and delivers, with shaping on. Total shaping?
Total reward for the walk, including the delivery step's step penalty?

<details><summary>Answer</summary>

Shaping `12 x 0.02 = 0.24`. Rewards: 12 steps at `-0.01` is `-0.12`, plus `+0.24` shaping, plus `+1.0`
on the last step: `1.12`. Without shaping it would be `0.88`. The shaping adds `0.02 x` the starting
distance, as [05](05-reward-shaping.md) explains.
</details>

## 10. Reward experiment (planned)

After Stage 3 and 5: set `COLLISION_PENALTY` to 0. Predict, then measure, the collision counts for
random, greedy and RL.

<details><summary>Answer</summary>

Random and greedy ignore rewards, so their collision counts do **not** change. RL will probably collide
more, because collisions no longer cost anything. Retrain RL, since the penalty is part of what it
learns from.
</details>

## 11. Find a map where greedy fails (planned)

After Stage 3: draw a small map with a wall between a robot and its target. Predict what greedy does,
then test it with a custom layout.

<details><summary>Answer</summary>

Greedy steps toward the target until the wall blocks it, then tries the other axis (zero offset on a
straight line), then sidesteps randomly. It will wander and may eventually find the way round, but
slowly and unreliably. See the dead-end example in [05](05-reward-shaping.md).
</details>

## 12. Parameter-sharing thought experiment

Three identical robots start in symmetric positions facing each other. With a deterministic shared
policy, what happens? Why does sampling help?

<details><summary>Answer</summary>

Identical weights plus identical (mirrored) observations give identical choices, so the robots move
the same way every step and can stay deadlocked forever. Sampling gives each robot an independent
random draw from the same probabilities, so their choices differ and the symmetry breaks.
</details>

## 13. Add path guidance (optional, planned)

After Stage 5: add the `distance_field` and `closer_flags` sketches from
[14](14-generalization.md) as four extra observation values and compare RL with and without them on
the *current* map and on one hand-drawn map with a dead end. Predict which wins on each before you run
it.

<details><summary>Answer</summary>

Expect little difference on the current map (detours are rare: at most 2 extra steps, 7.1% of pairs),
and a clear win for guidance on the dead-end map. If you see the opposite, check that the flags are
oriented correctly (up, down, left, right, in `ACTION_DELTAS` order).
</details>
