"""Sanity checks for the settings file, so an edited layout cannot silently break the game."""

from collections import deque

from multi_agent_warehouse.config import LAYOUT, MAX_ROBOTS


def free_cells() -> set[tuple[int, int]]:
    """Every `(row, col)` of floor in the layout."""
    return {(r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "."}


def test_layout_is_8_by_8():
    assert len(LAYOUT) == 8
    assert all(len(row) == 8 for row in LAYOUT)


def test_layout_uses_only_floor_and_shelf_characters():
    assert set("".join(LAYOUT)) <= {".", "#"}


def test_layout_has_room_for_every_robot_and_a_different_target():
    assert len(free_cells()) > MAX_ROBOTS


def test_all_free_cells_are_reachable_from_each_other():
    free = free_cells()
    start = next(iter(free))
    seen = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbour in free and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    assert seen == free, f"cells sealed off from the rest: {sorted(free - seen)}"
