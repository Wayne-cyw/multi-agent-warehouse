"""Pure move-resolution rules for the multi-agent warehouse.

Walls and shelves make a robot stay put (not a collision). Robots may not share a cell and may
not swap cells; a robot blocked by another robot stays, and then counts as a stayer.
"""

import numpy as np
from python_ta.contracts import check_contracts

Cell = tuple[int, int]

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# (row, col) offset for each action, indexed by action.
ACTION_DELTAS: tuple[Cell, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@check_contracts
def resolve_moves(
    positions: dict[str, Cell], actions: dict[str, int], shelves: np.ndarray
) -> tuple[dict[str, Cell], set[str]]:
    """Resolve simultaneous moves.

    Return the new position of every robot and the ids of robots that collided, meaning they
    tried to move and were blocked by another robot (directly or through a blocked robot ahead).
    Bumping the grid edge or a shelf just keeps the robot in place. Inputs are not mutated.

    Preconditions:
        - set(positions) == set(actions)
        - all(0 <= action < len(ACTION_DELTAS) for action in actions.values())
        - shelves.ndim == 2
        - every position lies inside the shelves grid
    """
    robot_ids = sorted(positions)
    proposals = {
        robot_id: _propose(positions[robot_id], actions[robot_id], shelves)
        for robot_id in robot_ids
    }
    moving = {robot_id for robot_id in robot_ids if proposals[robot_id] != positions[robot_id]}

    # A blocked robot becomes a stayer, which can block others in turn: repeat to a fixed point.
    # Each round judges every robot against the same snapshot, then blocks them all at once.
    while True:
        blocked = [
            robot_id
            for robot_id in robot_ids
            if _is_blocked(robot_id, robot_ids, positions, proposals)
        ]
        if not blocked:
            break
        for robot_id in blocked:
            proposals[robot_id] = positions[robot_id]

    collided = {robot_id for robot_id in moving if proposals[robot_id] == positions[robot_id]}
    return proposals, collided


def _propose(position: Cell, action: int, shelves: np.ndarray) -> Cell:
    """Return the cell a robot wants to enter, or its own cell if a wall or shelf is in the way.

    Preconditions:
        - 0 <= action < len(ACTION_DELTAS)
        - shelves.ndim == 2
    """
    height, width = shelves.shape
    d_row, d_col = ACTION_DELTAS[action]
    target = (position[0] + d_row, position[1] + d_col)
    in_bounds = 0 <= target[0] < height and 0 <= target[1] < width
    return target if in_bounds and not shelves[target] else position


def _is_blocked(
    robot_id: str, robot_ids: list[str], positions: dict[str, Cell], proposals: dict[str, Cell]
) -> bool:
    """Whether a moving robot shares its target cell with, or swaps cells with, another robot.

    Preconditions:
        - robot_id in robot_ids
        - set(robot_ids) == set(positions) == set(proposals)
    """
    if proposals[robot_id] == positions[robot_id]:
        return False
    for other in robot_ids:
        if other == robot_id:
            continue
        same_cell = proposals[other] == proposals[robot_id]
        swap = proposals[other] == positions[robot_id] and proposals[robot_id] == positions[other]
        if same_cell or swap:
            return True
    return False
