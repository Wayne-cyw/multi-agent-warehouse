# 11. Policies and the demo

[Index](README.md) · Previous: [10 rendering](10-rendering-pygame.md) · Next: [12 evaluation](12-evaluation.md)

> **Status:** implemented (Stage 3, Tasks 4 and 5): `agents.py` (random and greedy), `play.py`.
> `RLPolicy`, `available_policies()` and the `rl` demo mode were added in Stage 5 (Task 8).

## The policy interface

A **policy** here is any object with two methods:

```python
reset(seed: int | None = None) -> None
act(observations: dict[str, np.ndarray]) -> dict[str, int]
```

This is a `typing.Protocol`: *structural* typing, so anything with these two methods counts and no
base class is needed. Policies see **only observations**, never environment internals. That makes
comparisons fair (greedy gets no information a learned policy would not) and means the same
interface works for a neural network.

Because every policy has this interface, the demo, the evaluation and the RL agent can share one loop
([01](01-project-overview.md)).

## `RandomPolicy`

Picks uniformly from the five actions for each robot, using its own seeded RNG. It is the
**baseline**: any real policy should beat it.

## `GreedyPolicy`

A hand-written rule that uses the observation only:

1. Look at the target offset `(dr, dc)`. Step along the axis with the **larger absolute offset**
   (ties go to rows).
2. If that step is blocked, try the **other axis** if its offset is non-zero.
3. If still blocked, or the other offset is zero, pick a **random unblocked direction**.
4. If nothing is unblocked, or the robot is already on its target, **stay**.

Worked examples (from the plan's tests):

| Observation (dr, dc, blocked) | Action |
| --- | --- |
| `dc=0.5, dr=0.1` | RIGHT (column offset larger) |
| `dr=0.5, dc=0.1` | DOWN |
| `dr=0.1, dc=0.5`, right blocked | DOWN (other axis) |
| `dr=0, dc=0.5`, right and up blocked | DOWN or LEFT (only unblocked directions, random) |
| all four blocked | STAY |
| `dr = dc = 0` | STAY |

The **random sidestep** in step 3 is what gets greedy around shelves: a pure "step toward the target"
rule gets stuck against a shelf forever. It is not privileged, because it uses only the observation.

Greedy is a strong-ish baseline, which keeps any RL claim honest. Its blind spot is the same as a
learned policy's: it cannot see beyond adjacent cells, so a dead end would trap it
([14](14-generalization.md)). On the current map's open aisles that is rarely a problem.

## `RLPolicy`

Loads a saved PPO model and, each step:

1. stacks all robots' observations (sorted by robot id) into one batch;
2. calls `model.predict(batch, deterministic=False)` **once**;
3. maps the answers back to robots.

One network, one batch, all robots: parameter sharing at run time ([07](07-parameter-sharing-and-supersuit.md)).

`deterministic=False` means *sample* from the action probabilities. With identical weights and similar
observations, always taking the single most likely action can make robots mirror each other into a
standoff; sampling breaks the symmetry. It plays the same role the random sidestep plays for greedy.

`available_policies()` offers `rl` only when the model file exists and SB3 imports, so the project
works with or without the optional extras.

## The demo (`play.py`)

```bash
python play.py                          # greedy policy, window opens
python play.py --policy random --robots 4 --seed 3
python play.py --frames 30              # headless smoke test (stops after 30 frames)
```

| Item | Detail |
| --- | --- |
| Keys | **space** pause, **R** reset (same seed + 1), **P** cycle policy, **Esc / Q** quit |
| Flags | `--policy {random,greedy,rl}` (default `greedy`), `--robots` (3), `--seed` (0), `--fps` (10), `--frames` (0 = until quit) |
| Loop | handle keys, `actions = policy.act(obs)`, `env.step(actions)`, set `env.hud_note`, `env.render()` |
| End of episode | auto-resets with the next seed, and the policy is reset with it |
| Missing RL model | prints a one-line hint and carries on without the `rl` mode; it never crashes |

`--frames` exists so the demo can be smoke-tested without a display. `argparse`'s `choices` rejects an
unknown policy name for free.

## Check yourself

1. Why must policies use only observations?
2. What does the random sidestep fix in the greedy rule, and what does it still not fix?
3. In the greedy table, why is `dr=0.1, dc=0.5` with right blocked a DOWN move?
4. Why does `RLPolicy` call `predict` once on a batch?
5. What does the demo do when the RL model file is missing, and why?
