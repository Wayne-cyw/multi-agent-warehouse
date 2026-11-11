# 14. Generalization: making trained weights work on any layout

[Index](README.md) · Previous: [13 testing and tooling](13-testing-and-tooling.md) · Next: [15 glossary](15-glossary.md)

> **Status:** a design note, **not part of the current plan**. The project supports one fixed map
> and explicitly makes "no claim about generalization" a non-goal (D-03 and D-07 in
> [`../design-decisions.md`](../design-decisions.md)). Nothing in this note has been run through
> training, because no model exists yet. The code sketches *were* run against the real layout.

## The question

*If I change the warehouse layout, will the trained weights still work?*

**Short answer.** They will *load and run*, but they will probably perform worse, and sometimes much
worse. And "any layout" is a strong goal: no design guarantees it, but you can get a long way.

## What the weights actually know

The saved model maps **8 numbers to 5 action probabilities**. Neither the input nor the output depends
on the map, so the file loads on any layout whose observation has that shape. Nothing in the file
records which layout it was trained on. A mismatch is **silent**: it shows up as a lower score, never
an error.

What the 8 numbers contain, and how layout-specific each is:

| Index | Content | Layout-specific? |
| --- | --- | --- |
| 0, 1 | own row, own column, divided by `max(H, W) - 1` | **Yes.** Lets a network memorise "at (3,2), go right" for *this* map. Also its scale depends on grid size |
| 2, 3 | target offset, divided by `max(H, W) - 1` | **Partly.** The offset is egocentric, but the divisor changes with grid size: one cell is `1/7` here and `1/10` on an 11 x 11 map |
| 4 to 7 | blocked flags for up, down, left, right | No: purely local |

So two inputs leak the map, and the policy has **no memory and sees only adjacent cells**.

## What happens if you edit `LAYOUT` today

`config.py` says "Edit this file to change the game", and that is true for the *environment* and the
scripted policies. For trained weights:

| Change | Expected effect |
| --- | --- |
| Move a few shelves, same 8 x 8, similar open aisles | Mild drop. Local flags still guide it, but the memorised position cues are wrong in places |
| New shapes, narrow corridors or dead ends | Large drop. A no-memory local policy gets stuck (see below) |
| Different grid size | Distribution shift: the same offset maps to a different number. Also `test_layout_is_8_by_8` fails *by design* and the guard tests need updating |
| More robots than it trained with | It runs, but congestion is new |

These are expectations from the design, not measurements.

### Why dead ends are fatal to a reactive policy

A policy that sees only its four neighbours cannot know a dead end is coming. In the small map from
[05 reward shaping](05-reward-shaping.md), a robot at `(2,1)` with its target at `(2,3)` has a wall
between them: the right move is **up** (path length 4 down to 3), yet it *looks* like moving away
(Manhattan distance 2 up to 3). Greedy has the same blind spot. On the current map this barely
matters: the shortest path is never more than 2 longer than the Manhattan distance, and only 7.1% of
cell pairs need any detour *(computed on the 8 x 8 map)*. The aisles were designed that way. A new map need not be.

## A roadmap, cheapest and highest-value first

You do not need all of it. Each level is independent enough to try and measure.

### Level 0: measure the problem

Before changing anything, find out how bad it is. Make the layout selectable (for example an `env`
argument or an options key), then run the saved model on held-out maps and compare with random and
greedy using the same seeded evaluation ([12](12-evaluation.md)). **If RL scores below greedy on a new
map, it overfitted to the old one.**

### Level 1: remove what identifies the map

- **Drop absolute row and column** (indices 0 and 1). With nothing map-specific to memorise, the
  network can only use the offset and the local flags.
- **Do not divide the offset by grid size.** Use a fixed constant and clip to `[-1, 1]`, so "one cell
  away" is the same number on every map.

Cost: the observation shape changes (8 to 6), so old weights are invalid and tests need updating.
The robot can no longer tell where it is on the map, which matters little if the policy reacts to
local flags anyway. Revises D-11 and D-12.

### Level 2: train on many layouts

This is the most important change: no observation fix helps if the network only ever sees one map.

- Build the shelves and free cells **per episode**, for example through `reset(options={"layout": ...})`
  (PettingZoo's `reset` already accepts `options`; today `env.py` ignores it). `self.shelves`,
  `self._free_cells` and `self._scale` would become per-episode.
- Vary grid size, shelf density and robot count.
- **Check connectivity.** `tests/test_config.py` already contains the breadth-first search needed.
- Keep the observation **shape** fixed, which SuperSuit requires anyway ([07](07-parameter-sharing-and-supersuit.md)).

A practical warning *(computed)*: scattering shelves at random produces **disconnected maps most of
the time**. On 15 x 15 grids, about half of the random maps at 20% shelf density were fully connected,
**8 of 200** at 30%, and **none of 200** at 40%. (A rerun with independent shelf cells and a fixed seed gave the same trend, and on
8 x 8 grids it gave 132, 51 and 8 connected maps out of 200 at 20%, 30% and 40%; compare the trend, not
the digits.) So rejection sampling is wasteful above about 20%.
Prefer a structured generator (random placement of 2 x 2 or 2 x 3 shelf blocks with aisles, like the
current map) or carve a path to reconnect isolated cells.

Revises D-07 and D-03.

### Level 3: split the problem (path guidance)

A local view cannot see a dead end, but a **map-aware helper** can. Compute the shortest-path distance
to the target with a breadth-first search, then feed the policy, for each of the four neighbours,
whether stepping there gets it **closer along the real path**. The policy then only has to learn what
a map cannot tell it: **how to avoid the other robots**.

**Illustrative sketch** *(not project code; run against the real layout and the dead-end example)*:

```python
from collections import deque
import numpy as np
from multi_agent_warehouse.rules import ACTION_DELTAS

def distance_field(shelves, target):
    """Steps needed to reach `target` from every free cell (inf where unreachable)."""
    h, w = shelves.shape
    dist = np.full((h, w), np.inf)
    dist[target] = 0
    queue = deque([target])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS[1:]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not shelves[nr, nc] and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist

def closer_flags(dist, cell):
    """For up, down, left, right: +1 closer, 0 same or blocked, -1 farther (by path length)."""
    h, w = dist.shape
    flags = []
    for dr, dc in ACTION_DELTAS[1:]:
        nr, nc = cell[0] + dr, cell[1] + dc
        if not (0 <= nr < h and 0 <= nc < w) or not np.isfinite(dist[nr, nc]):
            flags.append(0.0)
        else:
            flags.append(float(np.sign(dist[cell] - dist[nr, nc])))
    return flags
```

Checked results: on the real map the distance from `(0,0)` to `(7,7)` is 14; at `(0,0)` the flags are
`[0, +1, 0, +1]` (down and right are closer, up and left are walls); and in the dead-end example the
flags at `(2,1)` are `[+1, -1, 0, 0]`: *up is closer* while Manhattan distance says it is farther.

Two companions:

- **Switch the shaping potential to this path distance** (the same BFS), so shaping stops punishing
  the right move in a detour ([05](05-reward-shaping.md)). Revises D-21.
- **Upgrade the greedy baseline** to use the same guidance (a "BFS greedy"), otherwise comparing RL
  with plain greedy on hard maps is unfair.

Trade-off: the robot is assumed to **know the map**. That is realistic for warehouse robots (a global
route planner plus local collision avoidance is a standard split), and it is close to what
multi-agent path-finding research does with a local view plus a goal direction (for example PRIMAL;
see [17](17-further-reading.md), from memory). It is not "learning navigation from scratch".

### Level 4: see further, or remember

If guidance is not allowed or not enough:

| Option | What it adds | Cost |
| --- | --- | --- |
| **Wider local window** (5 x 5 or 7 x 7 patch of shelves and robots) | Sees around corners | A bigger observation; still blind beyond its radius |
| **Memory** (a recurrent policy) | Remembers where it has been | Slower, harder training |

### Level 5: an honest evaluation protocol

- Keep **train**, **validation** and **test** layouts separate, and include hand-built stress maps
  (corridors, rooms, dead ends).
- Report **in-distribution and held-out** results separately. A big gap means overfitting.
- Compare with random and the upgraded greedy. Fix the success criterion **before** looking at the
  numbers, as the project already does for the single-layout bar ([12](12-evaluation.md)).
- Vary the number of robots too.

## What would still go wrong

"Any layout" cannot be promised:

- **Single-width corridors with several robots** can deadlock. Two robots meeting head-on need to
  *coordinate* (one yields). A local reactive policy has only the stay action and luck.
- **Very different scales** (a 50 x 50 map) need longer episodes and a different step budget, and the
  reward magnitudes may need retuning.
- **Dense traffic** is a different regime from three robots on an open map.
- **Non-grid worlds** are a different environment altogether.

## Which decisions this would revise

| Decision | Today | Under generalization |
| --- | --- | --- |
| D-03 | One layout; no generalisation claim | Several layouts; a measured generalisation claim |
| D-07 | One hard-coded map in `config.py` | Layouts chosen or generated per episode |
| D-11 | 8-number observation with absolute position | Fewer map-identifying inputs; optionally path-guidance features or a window |
| D-12 | Divide by `max(H, W) - 1` | Fixed constant, clipped |
| D-21 | Manhattan shaping | Path-distance shaping |
| D-35, D-44 | One-layout evaluation and acceptance bar | Held-out layout evaluation |

## Recommended order

1. Finish Stages 1 to 5 on the **single layout**. That gives you a measured baseline, and it answers
   the biggest technical question first: does the SuperSuit and SB3 stack work at all
   ([07](07-parameter-sharing-and-supersuit.md))?
2. Then add generalization as a separate **Stage 6**, one level at a time, measuring after each.

A sensible Stage 6 gate: after Level 1 and Level 2, does RL on **held-out** maps still beat random by
the same 2x margin the single-layout bar uses, measured on those same held-out maps? If not, add
Level 3 before anything fancier. This is tracked as decision D-53 in
[`../design-decisions.md`](../design-decisions.md).

## Check yourself

1. Why does the saved model load on a new layout without any error?
2. Which two observation values let a network memorise the map, and what does dropping them cost?
3. Why does changing the grid size change the *meaning* of the offset inputs?
4. Why is training on many layouts more important than any observation change?
5. Why are randomly scattered shelves a poor way to generate layouts?
6. What does the distance-field helper give the policy, and what is the cost of that design?
7. Name one layout feature that no local reactive policy will handle well.
