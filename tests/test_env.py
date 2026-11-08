"""Tests for the multi-agent warehouse ParallelEnv."""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo.test import parallel_api_test

from multi_agent_warehouse import MultiAgentWarehouseEnv
from multi_agent_warehouse.agents import GreedyPolicy, RandomPolicy
from multi_agent_warehouse.config import LAYOUT
from multi_agent_warehouse.evaluate import run_episode
from multi_agent_warehouse.rules import DOWN, RIGHT, STAY, UP

R0, R1, R2 = "robot_0", "robot_1", "robot_2"
# Index of the last row and column; also the observation scale.
LAST = len(LAYOUT) - 1
MID = LAST // 2


def make_env(**kwargs) -> MultiAgentWarehouseEnv:
    """Build and reset an env, then park robots 1 and 2 far from the action.

    Robot 1 sits in the bottom-right corner and robot 2 in the bottom-left, each with a target on
    the middle row that a parked robot never reaches, so tests only see the robots they move on
    purpose. The tests also rely on the corners and the whole of rows 0, 3 and `MID` being free.
    """
    env = MultiAgentWarehouseEnv(**kwargs)
    env.reset(seed=0)
    env.positions = {R0: (0, 0), R1: (LAST, LAST), R2: (LAST, 0)}
    env.targets = {R0: (LAST, 3), R1: (MID, 0), R2: (MID, LAST)}
    return env


def stay_except(**moves: int) -> dict[str, int]:
    """Actions where every robot stays unless named in `moves` (e.g. robot_0=RIGHT)."""
    actions = {R0: STAY, R1: STAY, R2: STAY}
    actions.update(moves)
    return actions


def test_parallel_api():
    parallel_api_test(MultiAgentWarehouseEnv(), num_cycles=300)


def test_spaces():
    env = MultiAgentWarehouseEnv()
    obs_space = env.observation_space("robot_0")
    assert isinstance(obs_space, Box)
    assert obs_space.shape == (8,)
    assert obs_space.dtype == np.float32
    assert (obs_space.low == -1).all() and (obs_space.high == 1).all()
    assert env.action_space("robot_0") == Discrete(5)


def test_metadata_and_defaults():
    env = MultiAgentWarehouseEnv()
    assert env.metadata["name"] == "multi_agent_warehouse_v0"
    assert env.possible_agents == ["robot_0", "robot_1", "robot_2"]
    assert env.hud_note == ""
    assert env.render() is None
    env.close()


def test_reset_places_robots_on_distinct_free_cells_with_valid_targets():
    env = MultiAgentWarehouseEnv(n_robots=6)
    for seed in range(20):
        env.reset(seed=seed)
        cells = list(env.positions.values())
        assert len(set(cells)) == len(cells) == 6
        for agent in env.agents:
            assert not env.shelves[env.positions[agent]]
            assert not env.shelves[env.targets[agent]]
            assert env.targets[agent] != env.positions[agent]


def test_same_seed_same_trajectory():
    def run(seed: int) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Play a fixed-length random episode and return its observations and rewards."""
        env = MultiAgentWarehouseEnv()
        obs, _ = env.reset(seed=seed)
        observations = [np.concatenate(list(obs.values()))]
        rewards_trace = []
        action_rng = np.random.default_rng(0)
        for _ in range(50):
            actions = {a: int(action_rng.integers(5)) for a in env.agents}
            obs, rewards, *_ = env.step(actions)
            observations.append(np.concatenate(list(obs.values())))
            rewards_trace.append(rewards)
        return np.array(observations), rewards_trace

    obs_a, rewards_a = run(7)
    obs_b, rewards_b = run(7)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert rewards_a == rewards_b
    assert not np.array_equal(obs_a, run(8)[0])


def test_reset_without_seed_continues_the_stream():
    env = MultiAgentWarehouseEnv()
    env.reset(seed=3)
    first = dict(env.positions)
    env.reset()
    second = dict(env.positions)
    env.reset(seed=3)
    assert dict(env.positions) == first
    env.reset()
    assert dict(env.positions) == second


def test_delivery():
    env = make_env()
    env.positions[R0] = (0, 0)
    env.targets[R0] = (0, 1)
    obs, rewards, _, _, infos = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(1.0 - 0.01)
    assert infos[R0]["deliveries"] == 1
    assert infos[R0]["collided"] is False
    new_target = env.targets[R0]
    assert not env.shelves[new_target]
    assert new_target != (0, 1)
    assert env.total_deliveries == 1
    # The observation already shows the new target.
    assert obs[R0][2] == pytest.approx((new_target[0] - 0) / LAST)
    assert obs[R0][3] == pytest.approx((new_target[1] - 1) / LAST)


def test_collision_reward():
    env = make_env()
    env.positions[R0] = (0, 0)
    env.positions[R1] = (0, 2)
    env.targets[R0] = (LAST, 3)
    env.targets[R1] = (LAST, 3)
    _, rewards, _, _, infos = env.step(stay_except(robot_0=RIGHT, robot_1=3))
    for agent in (R0, R1):
        assert rewards[agent] == pytest.approx(-0.01 - 0.1)
        assert infos[agent]["collided"] is True
    assert rewards[R2] == pytest.approx(-0.01)
    assert infos[R2]["collided"] is False
    assert env.total_collisions == 2


def test_bump_has_no_penalty():
    env = make_env()
    _, rewards, _, _, infos = env.step(stay_except(robot_0=UP))
    assert rewards[R0] == pytest.approx(-0.01)
    assert infos[R0]["collided"] is False
    assert env.positions[R0] == (0, 0)
    assert env.total_collisions == 0


def test_observation_values():
    env = make_env()
    env.targets[R0] = (LAST, LAST)
    # Not adjacent to (0, 0).
    env.positions[R1] = (LAST, LAST - 1)
    obs, *_ = env.step(stay_except())
    assert obs[R0].dtype == np.float32
    np.testing.assert_allclose(obs[R0], [0, 0, 1, 1, 1, 0, 1, 0])


def test_shelf_sets_blocked_flag():
    env = make_env()
    # The shelf at (1, 1) is directly below.
    env.positions[R0] = (0, 1)
    obs, *_ = env.step(stay_except())
    # Down is blocked by the shelf and up by the grid edge.
    assert obs[R0][5] == 1.0
    assert obs[R0][4] == 1.0
    assert obs[R0][6] == 0.0 and obs[R0][7] == 0.0


def test_neighbour_flag_for_other_robot():
    env = make_env()
    env.positions[R0] = (0, 0)
    env.positions[R1] = (0, 1)
    obs, *_ = env.step(stay_except())
    assert obs[R0][7] == 1.0
    assert obs[R1][6] == 1.0


def test_shaping():
    env = make_env(shaping=True)
    env.targets[R0] = (0, 3)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(-0.01 + 0.02)

    env = make_env()
    env.targets[R0] = (0, 3)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(-0.01)


def test_shaping_moving_away_is_negative():
    env = make_env(shaping=True)
    env.positions[R0] = (3, 3)
    env.targets[R0] = (3, 0)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(-0.01 - 0.02)


def test_shaping_on_delivery_uses_old_target():
    env = make_env(shaping=True)
    env.targets[R0] = (0, 1)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(1.0 - 0.01 + 0.02)


def test_truncates_at_max_steps():
    env = MultiAgentWarehouseEnv(max_steps=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, terminations, truncations, _ = env.step({a: STAY for a in env.agents})
        assert not any(terminations.values())
        expected = i == 4
        assert all(v is expected for v in truncations.values())
    assert env.agents == []
    assert env.step_count == 5


def test_step_after_end_returns_empty():
    env = MultiAgentWarehouseEnv(max_steps=1)
    env.reset(seed=0)
    env.step({a: STAY for a in env.agents})
    assert env.agents == []
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_invalid_actions_raise():
    env = MultiAgentWarehouseEnv()
    env.reset(seed=0)
    good = {a: STAY for a in env.agents}
    bad_cases = [
        # A missing key, then an extra key.
        {k: v for k, v in good.items() if k != R2},
        {**good, "robot_9": STAY},
        {**good, R0: 5},
        {**good, R0: -1},
        {**good, R0: 1.5},
        {**good, R0: "a"},
    ]
    before = (dict(env.positions), dict(env.targets), env.step_count)
    for actions in bad_cases:
        with pytest.raises(AssertionError):
            env.step(actions)
    assert (env.positions, env.targets, env.step_count) == before
    # numpy integers are fine.
    env.step({a: np.int64(DOWN) for a in env.agents})


def test_invalid_constructor_args_raise():
    for kwargs in ({"n_robots": 0}, {"n_robots": 7}, {"render_mode": "bogus"}, {"max_steps": 0}):
        with pytest.raises(ValueError):
            MultiAgentWarehouseEnv(**kwargs)


# Results of the default env before reward parameters were added; they must never drift.
PINNED_GREEDY = {"deliveries": 18, "collisions": 338}
PINNED_RANDOM = {"deliveries": 3, "collisions": 5}


def test_default_episodes_match_pinned_results():
    assert run_episode(MultiAgentWarehouseEnv(), GreedyPolicy(), 1000) == PINNED_GREEDY
    assert run_episode(MultiAgentWarehouseEnv(), RandomPolicy(), 1000) == PINNED_RANDOM


def test_custom_collision_penalty_is_used():
    env = make_env(collision_penalty=-0.5)
    env.positions[R0] = (0, 0)
    env.positions[R1] = (0, 2)
    env.targets[R0] = (LAST, 3)
    env.targets[R1] = (LAST, 3)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT, robot_1=3))
    assert rewards[R0] == pytest.approx(-0.01 - 0.5)
    assert rewards[R1] == pytest.approx(-0.01 - 0.5)
    assert rewards[R2] == pytest.approx(-0.01)


def test_custom_shaping_scale_is_used():
    env = make_env(shaping=True, shaping_scale=0.1)
    env.targets[R0] = (0, 3)
    _, rewards, *_ = env.step(stay_except(robot_0=RIGHT))
    assert rewards[R0] == pytest.approx(-0.01 + 0.1)


def test_truncation_flag_only_on_last_step():
    env = MultiAgentWarehouseEnv()
    env.reset(seed=0)
    actions = dict.fromkeys(env.agents, STAY)
    for _ in range(env.max_steps - 1):
        _, _, _, _, infos = env.step(actions)
        assert all(info["TimeLimit.truncated"] is False for info in infos.values())
    _, _, _, truncations, infos = env.step(actions)
    assert all(truncations.values())
    assert all(info["TimeLimit.truncated"] is True for info in infos.values())


def test_invalid_reward_parameters_raise():
    for kwargs in ({"collision_penalty": 0.1}, {"shaping_scale": -0.1}):
        with pytest.raises(ValueError):
            MultiAgentWarehouseEnv(**kwargs)


def test_rich_observation_has_14_values_and_basic_prefix():
    basic = make_env()
    rich = make_env(observation="rich")
    assert rich.observation_space(R0).shape == (14,)
    assert basic.observation_space(R0).shape == (8,)
    basic_obs, *_ = basic.step(stay_except())
    rich_obs, *_ = rich.step(stay_except())
    for agent in rich.agents:
        assert rich_obs[agent].shape == (14,)
        assert rich_obs[agent].dtype == np.float32
        np.testing.assert_array_equal(rich_obs[agent][:8], basic_obs[agent])
        assert rich.observation_space(agent).contains(rich_obs[agent])


def test_rich_robot_flags_separate_robots_from_shelves():
    env = make_env(observation="rich")
    env.positions[R0] = (0, 1)
    env.positions[R1] = (0, 2)
    obs, *_ = env.step(stay_except())
    # The shelf below robot 0 sets its basic down flag but not its robot flag.
    assert obs[R0][5] == 1.0 and obs[R0][9] == 0.0
    # Robot 1 is to the right of robot 0: both flags are set.
    assert obs[R0][7] == 1.0 and obs[R0][11] == 1.0
    assert obs[R1][6] == 1.0 and obs[R1][10] == 1.0
    assert obs[R0][8] == 0.0 and obs[R0][10] == 0.0


def test_nearest_robot_offset_and_tie_break():
    env = make_env(observation="rich")
    env.positions = {R0: (3, 3), R1: (3, 6), R2: (6, 3)}
    obs, *_ = env.step(stay_except())
    # Robots 1 and 2 are both 3 away from robot 0, so the lower id wins.
    np.testing.assert_allclose(obs[R0][12:], [0, 3 / LAST])
    # For robot 1, robot 0 is 3 away and robot 2 is 6 away.
    np.testing.assert_allclose(obs[R1][12:], [0, -3 / LAST])
    env.positions = {R0: (0, 0), R1: (LAST, LAST), R2: (0, 2)}
    obs, *_ = env.step(stay_except())
    np.testing.assert_allclose(obs[R0][12:], [0, 2 / LAST])
    np.testing.assert_allclose(obs[R1][12:], [-LAST / LAST, -(LAST - 2) / LAST])


def test_single_robot_rich_offset_is_zero():
    env = MultiAgentWarehouseEnv(n_robots=1, observation="rich")
    obs, _ = env.reset(seed=0)
    assert obs[R0].shape == (14,)
    np.testing.assert_array_equal(obs[R0][8:], np.zeros(6, dtype=np.float32))


def test_invalid_observation_mode_raises():
    with pytest.raises(ValueError, match="basic"):
        MultiAgentWarehouseEnv(observation="bogus")


def test_rich_parallel_api():
    parallel_api_test(MultiAgentWarehouseEnv(observation="rich"), num_cycles=300)


def test_default_observation_is_basic_and_unchanged():
    env = MultiAgentWarehouseEnv()
    assert env.observation == "basic"
    obs, _ = env.reset(seed=0)
    assert all(o.shape == (8,) for o in obs.values())
    explicit = MultiAgentWarehouseEnv(observation="basic")
    explicit_obs, _ = explicit.reset(seed=0)
    for agent in obs:
        np.testing.assert_array_equal(obs[agent], explicit_obs[agent])
    pinned = run_episode(MultiAgentWarehouseEnv(observation="basic"), GreedyPolicy(), 1000)
    assert pinned == PINNED_GREEDY
