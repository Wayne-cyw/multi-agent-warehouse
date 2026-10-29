"""Tests for the pygame renderer and `env.render`."""

import numpy as np

from multi_agent_warehouse import MultiAgentWarehouseEnv
from multi_agent_warehouse.config import LAYOUT
from multi_agent_warehouse.render import (
    CELL,
    FLOOR_COLOR,
    HUD_H,
    ROBOT_COLORS,
    SHELF_COLOR,
    TARGET_INSET,
    Renderer,
)

ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
SHELVES = np.array([[ch == "#" for ch in row] for row in LAYOUT], dtype=bool)
HUD_LINES = ["step 0", "deliveries 0", "collisions 0", ""]


def pixel(frame: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    """The RGB value at pixel (x, y) of a frame."""
    return tuple(int(v) for v in frame[y, x])


def test_draw_shape_and_dtype():
    renderer = Renderer(SHELVES)
    frame = renderer.draw({}, {}, HUD_LINES)
    renderer.close()
    assert frame.shape == (HUD_H + ROWS * CELL, COLS * CELL, 3)
    assert frame.dtype == np.uint8


def test_pixels():
    renderer = Renderer(SHELVES)
    positions = {"robot_0": (0, 0), "robot_1": (0, 1)}
    targets = {"robot_0": (0, 3), "robot_1": (3, 3)}
    frame = renderer.draw(positions, targets, HUD_LINES)

    x, y = renderer.cell_origin(0, 0)
    assert pixel(frame, x + CELL // 2, y + CELL // 2) == ROBOT_COLORS[0]

    x, y = renderer.cell_origin(1, 1)
    assert LAYOUT[1][1] == "#"
    assert pixel(frame, x + CELL // 2, y + CELL // 2) == SHELF_COLOR

    x, y = renderer.cell_origin(0, 3)
    assert pixel(frame, x + TARGET_INSET + 2, y + CELL // 2) == ROBOT_COLORS[0]

    x, y = renderer.cell_origin(4, 4)
    assert LAYOUT[4][4] == "."
    assert pixel(frame, x + 2, y + 2) == FLOOR_COLOR
    renderer.close()


def test_cell_origin_includes_hud_offset():
    renderer = Renderer(SHELVES)
    assert renderer.cell_origin(0, 0) == (0, HUD_H)
    assert renderer.cell_origin(2, 3) == (3 * CELL, HUD_H + 2 * CELL)
    renderer.close()


def test_env_rgb_array_render():
    env = MultiAgentWarehouseEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame.shape == (HUD_H + ROWS * CELL, COLS * CELL, 3)
    assert frame.dtype == np.uint8


def test_env_human_render_returns_none():
    env = MultiAgentWarehouseEnv(render_mode="human")
    env.reset(seed=0)
    assert env.render() is None
    env.close()


def test_render_mode_none_returns_none():
    env = MultiAgentWarehouseEnv()
    env.reset(seed=0)
    assert env.render() is None
    assert env._renderer is None
    env.close()


def test_close_twice_is_safe():
    env = MultiAgentWarehouseEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()
    renderer = Renderer(SHELVES)
    renderer.close()
    renderer.close()


def test_close_without_render_is_safe():
    env = MultiAgentWarehouseEnv(render_mode="rgb_array")
    env.close()
    env.close()


def test_rendering_does_not_change_trajectory():
    plain = MultiAgentWarehouseEnv(render_mode=None)
    drawn = MultiAgentWarehouseEnv(render_mode="rgb_array")
    obs_a, _ = plain.reset(seed=7)
    obs_b, _ = drawn.reset(seed=7)
    drawn.render()
    action_rng = np.random.default_rng(3)
    for _ in range(30):
        actions = {a: int(action_rng.integers(5)) for a in plain.agents}
        obs_a, rew_a, *_ = plain.step(actions)
        obs_b, rew_b, *_ = drawn.step(actions)
        drawn.render()
        assert rew_a == rew_b
        for agent in obs_a:
            np.testing.assert_array_equal(obs_a[agent], obs_b[agent])
    drawn.close()


def test_hud_note_changes_the_frame():
    env = MultiAgentWarehouseEnv(render_mode="rgb_array")
    env.reset(seed=0)
    before = env.render()
    env.hud_note = "PAUSED"
    after = env.render()
    env.close()
    assert not np.array_equal(before[:HUD_H], after[:HUD_H])
    np.testing.assert_array_equal(before[HUD_H:], after[HUD_H:])
