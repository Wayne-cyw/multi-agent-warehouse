"""Pure move-resolution rules for the mini warehouse.

Walls and shelves make a robot stay put (not a collision). Robots may not share a cell and may
not swap cells; a robot blocked by another robot stays, and then counts as a stayer.
"""

import numpy as np

Cell = tuple[int, int]

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# (row, col) offset for each action, indexed by action.
ACTION_DELTAS: tuple[Cell, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _is_blocked(
    rid: str, ids: list[str], positions: dict[str, Cell], proposals: dict[str, Cell]
) -> bool:
    """Whether a moving robot shares its target cell with, or swaps cells with, another robot."""
    if proposals[rid] == positions[rid]:
        return False
    for other in ids:
        if other == rid:
            continue
        same_cell = proposals[other] == proposals[rid]
        swap = proposals[other] == positions[rid] and proposals[rid] == positions[other]
        if same_cell or swap:
            return True
    return False


def resolve_moves(
    positions: dict[str, Cell], actions: dict[str, int], shelves: np.ndarray
) -> tuple[dict[str, Cell], set[str]]:
    """Resolve simultaneous moves.

    Returns the new position of every robot and the ids of robots that collided, meaning they
    tried to move and were blocked by another robot (directly or through a blocked robot ahead).
    Bumping the grid edge or a shelf just keeps the robot in place. Inputs are not mutated.
    """
    height, width = shelves.shape
    ids = sorted(positions)

    proposals: dict[str, Cell] = {}
    for rid in ids:
        row, col = positions[rid]
        d_row, d_col = ACTION_DELTAS[actions[rid]]
        target = (row + d_row, col + d_col)
        in_bounds = 0 <= target[0] < height and 0 <= target[1] < width
        proposals[rid] = target if in_bounds and not shelves[target] else positions[rid]

    moving = {rid for rid in ids if proposals[rid] != positions[rid]}

    # A blocked robot becomes a stayer, which can block others in turn: repeat to a fixed point.
    # Each round judges every robot against the same snapshot, then blocks them all at once.
    while True:
        blocked = [rid for rid in ids if _is_blocked(rid, ids, positions, proposals)]
        if not blocked:
            break
        for rid in blocked:
            proposals[rid] = positions[rid]

    collided = {rid for rid in moving if proposals[rid] == positions[rid]}
    return proposals, collided
