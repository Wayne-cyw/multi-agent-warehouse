# 09. Move resolution

[Index](README.md) · Previous: [08 warehouse rules](08-warehouse-rules.md) · Next: [10 rendering](10-rendering-pygame.md)

All robots move **simultaneously**, so the environment must decide what happens on conflicts. That is
the job of `resolve_moves` in `multi_agent_warehouse/rules.py` *(verified)*: a **pure function** (no
env state, inputs are not mutated) that returns the new position of every robot and the set of robots
that **collided**.

```python
resolve_moves(positions: dict[str, Cell], actions: dict[str, int], shelves: np.ndarray)
    -> tuple[dict[str, Cell], set[str]]
```

## The algorithm

**Step A: proposals.** Each robot proposes `position + delta`. If that is out of bounds or a shelf,
the proposal is replaced by the robot's own cell (it "bumps" and stays).

**Step B: two rules, repeated until nothing changes** (a *fixed point*):

1. **Same cell**: a moving robot is blocked if another robot proposes the **same** cell. A robot that
   stays proposes its own cell, so "moving into a stayer" is just this rule.
2. **Swap**: two robots exchanging cells are both blocked.

A blocked robot stays put, so it now counts as a **stayer**, which may block someone behind it. That
is why the rules repeat: blocking can **cascade**.

**Collision** = a robot that *tried to move* and ended up blocked by another robot, directly or
through the cascade. A **wall or shelf bump is not a collision**, and a robot that chose to stay
never collides.

## The real code

```python
def _is_blocked(rid, ids, positions, proposals) -> bool:
    if proposals[rid] == positions[rid]:
        return False                      # a stayer is never "blocked"
    for other in ids:
        if other == rid:
            continue
        same_cell = proposals[other] == proposals[rid]
        swap = proposals[other] == positions[rid] and proposals[rid] == positions[other]
        if same_cell or swap:
            return True
    return False

# inside resolve_moves, after building `proposals` and `moving`:
while True:
    blocked = [rid for rid in ids if _is_blocked(rid, ids, positions, proposals)]
    if not blocked:
        break
    for rid in blocked:
        proposals[rid] = positions[rid]

collided = {rid for rid in moving if proposals[rid] == positions[rid]}
return proposals, collided
```

`moving` is the set of robots whose proposal differed from their position *before* any blocking, so a
robot that was blocked is in `moving` and ends at its own cell, which is exactly "tried to move and
did not".

## Worked examples

Each is a real test case in `tests/test_rules.py` and was re-run against the real function *(verified)*.
`->` marks the move; `.` is empty floor; coordinates are `(row, col)`.

```
two into one          a(2,1)->    <-b(2,3)       both want (2,2)
   . A -> . <- B .                               both blocked, both collide

into a stayer         a(2,1)->    b(2,2) stays
   . A -> B . . .                                a blocked and collides; b is fine

swap                  a(2,1)->    <-b(2,2)
   . A -> <- B . .                               both blocked, both collide

follow the leader     a(2,1)->    b(2,2)->
   . A -> B -> . .                               a ends at (2,2), b at (2,3): no collision

cascade               a(2,0)->  b(2,1)->  <-c(2,3)
   A -> B -> . <- C                              round 1: b and c both want (2,2): blocked.
                                                 round 2: b is now a stayer, so a is blocked.
                                                 a, b, c all collide.

behind a wall bump    shelf at (2,3);  a(2,1)->  b(2,2)->
   . A -> B -> # . .                             b bumps the shelf and stays (no collision).
                                                 a is now moving into a stayer: a collides.

rotation (allowed)    2x2 block, a->  b v  c <-  d ^   (clockwise)
   A -> B                                        nobody ends on a cell another robot
   ^    v                                        proposes, no swap, so all four move.
   D <- C
```

Two deliberate simplifications (see D-15 and D-16 in [`../design-decisions.md`](../design-decisions.md)):

- A rotation of four robots around a 2 x 2 block is **allowed**. Physically it would be a collision,
  but catching it needs cycle detection for little benefit.
- A robot **held up behind** a conflict also gets the collision penalty.

## A bug worth learning from

An early prototype of this loop changed each robot's proposal *inside* the loop that was still
checking other robots. For "two into one" it blocked `a`, then checked `b` against `a`'s
**already-reset** proposal, found no conflict, and let `b` through. Three of the eleven test cases
failed.

The fix, and what the real code does, is **decide every block in a round from the same snapshot, then
apply them all together**: the list comprehension builds `blocked` completely *before* any proposal is
changed. Lesson: when many things change "at once", never mutate state you are still reading. A useful
side effect: the outcome no longer depends on iteration order, though the code still iterates in
sorted-id order (`ids = sorted(positions)`) for determinism.

## Why a pure function

- It can be tested **without constructing an environment**: pass in dicts and an array, check the
  result.
- Physics and PettingZoo plumbing stay separate, so the env reads as a pipeline.
- It never mutates its inputs (a test pins this: `test_inputs_not_mutated`).

## Check yourself

1. Why does the algorithm repeat until nothing changes? Give a three-robot example.
2. In "behind a wall bump", why does `b` not collide but `a` does?
3. Why is the rotation allowed, and what would be needed to forbid it?
4. Two robots at `(2,2)` and `(2,3)` both move right, with free floor beyond. What happens?
5. What was the bug in the first prototype, and how does the real code avoid it?
6. Why is a robot that chose `STAY` never counted as collided?
