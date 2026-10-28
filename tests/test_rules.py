"""Tests for the pure move-resolution rules of the multi-agent warehouse."""

import numpy as np

from multi_agent_warehouse.rules import DOWN, LEFT, RIGHT, STAY, UP, Cell, resolve_moves


def resolve(
    pos: dict[str, Cell], acts: dict[str, int], shelves: np.ndarray | None = None
) -> tuple[dict[str, Cell], set[str]]:
    """Resolve moves on a 5 x 5 grid, with no shelves unless given."""
    if shelves is None:
        shelves = np.zeros((5, 5), dtype=bool)
    return resolve_moves(pos, acts, shelves)


def test_free_move():
    new, collided = resolve({"a": (2, 2)}, {"a": RIGHT})
    assert new == {"a": (2, 3)}
    assert collided == set()


def test_wall_and_shelf_bump_are_not_collisions():
    new, collided = resolve({"a": (0, 0)}, {"a": UP})
    assert new == {"a": (0, 0)}
    assert collided == set()

    shelves = np.zeros((5, 5), dtype=bool)
    shelves[2, 3] = True
    new, collided = resolve({"a": (2, 2)}, {"a": RIGHT}, shelves)
    assert new == {"a": (2, 2)}
    assert collided == set()


def test_two_into_one():
    pos = {"a": (2, 1), "b": (2, 3)}
    new, collided = resolve(pos, {"a": RIGHT, "b": LEFT})
    assert new == pos
    assert collided == {"a", "b"}


def test_three_into_one():
    pos = {"a": (2, 1), "b": (2, 3), "c": (1, 2)}
    new, collided = resolve(pos, {"a": RIGHT, "b": LEFT, "c": DOWN})
    assert new == pos
    assert collided == {"a", "b", "c"}


def test_into_a_stayer():
    pos = {"a": (2, 1), "b": (2, 2)}
    new, collided = resolve(pos, {"a": RIGHT, "b": STAY})
    assert new == pos
    assert collided == {"a"}


def test_swap():
    pos = {"a": (2, 1), "b": (2, 2)}
    new, collided = resolve(pos, {"a": RIGHT, "b": LEFT})
    assert new == pos
    assert collided == {"a", "b"}


def test_follow_the_leader():
    new, collided = resolve({"a": (2, 1), "b": (2, 2)}, {"a": RIGHT, "b": RIGHT})
    assert new == {"a": (2, 2), "b": (2, 3)}
    assert collided == set()


def test_cascade():
    pos = {"a": (2, 0), "b": (2, 1), "c": (2, 3)}
    new, collided = resolve(pos, {"a": RIGHT, "b": RIGHT, "c": LEFT})
    assert new == pos
    assert collided == {"a", "b", "c"}


def test_behind_a_wall_bump():
    shelves = np.zeros((5, 5), dtype=bool)
    shelves[2, 3] = True
    pos = {"a": (2, 1), "b": (2, 2)}
    new, collided = resolve(pos, {"a": RIGHT, "b": RIGHT}, shelves)
    assert new == pos
    assert collided == {"a"}


def test_rotation_is_allowed():
    pos = {"a": (1, 1), "b": (1, 2), "c": (2, 2), "d": (2, 1)}
    acts = {"a": RIGHT, "b": DOWN, "c": LEFT, "d": UP}
    new, collided = resolve(pos, acts)
    assert new == {"a": (1, 2), "b": (2, 2), "c": (2, 1), "d": (1, 1)}
    assert collided == set()


def test_inputs_not_mutated():
    pos = {"a": (2, 1), "b": (2, 3), "c": (1, 2)}
    acts = {"a": RIGHT, "b": LEFT, "c": DOWN}
    shelves = np.zeros((5, 5), dtype=bool)
    shelves[0, 0] = True
    pos_before, acts_before, shelves_before = dict(pos), dict(acts), shelves.copy()
    resolve_moves(pos, acts, shelves)
    assert pos == pos_before
    assert acts == acts_before
    assert np.array_equal(shelves, shelves_before)
