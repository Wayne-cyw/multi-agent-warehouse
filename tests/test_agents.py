"""Tests for the scripted policies and the evaluation loop."""

import numpy as np

from multi_agent_warehouse import MultiAgentWarehouseEnv
from multi_agent_warehouse.agents import POLICIES, GreedyPolicy, RandomPolicy
from multi_agent_warehouse.evaluate import evaluate, run_episode
from multi_agent_warehouse.rules import DOWN, LEFT, RIGHT, STAY, UP

R0 = "robot_0"


def obs(dr: float, dc: float, up=0, down=0, left=0, right=0) -> dict[str, np.ndarray]:
    """One robot's observation: position zeroed, target offset and blocked flags as given."""
    vector = np.array([0, 0, dr, dc, up, down, left, right], dtype=np.float32)
    return {R0: vector}


def greedy(seed: int = 0) -> GreedyPolicy:
    """A greedy policy reset with `seed`."""
    policy = GreedyPolicy()
    policy.reset(seed)
    return policy


def test_random_policy_valid_and_seeded():
    observations = {f"robot_{i}": np.zeros(8, dtype=np.float32) for i in range(3)}
    first, second = RandomPolicy(), RandomPolicy()
    first.reset(3)
    second.reset(3)
    for _ in range(20):
        actions = first.act(observations)
        assert set(actions) == set(observations)
        assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions.values())
        assert actions == second.act(observations)


def test_greedy_moves_along_larger_axis():
    assert greedy().act(obs(dr=0.1, dc=0.5))[R0] == RIGHT
    assert greedy().act(obs(dr=0.5, dc=0.1))[R0] == DOWN
    assert greedy().act(obs(dr=-0.5, dc=0.1))[R0] == UP
    assert greedy().act(obs(dr=0.1, dc=-0.5))[R0] == LEFT
    # A tie goes to rows.
    assert greedy().act(obs(dr=0.3, dc=0.3))[R0] in (DOWN, UP)
    assert greedy().act(obs(dr=-0.3, dc=0.3))[R0] == UP


def test_greedy_tries_other_axis_when_blocked():
    assert greedy().act(obs(dr=0.1, dc=0.5, right=1))[R0] == DOWN


def test_greedy_sidesteps_when_other_axis_is_zero():
    policy = greedy()
    for _ in range(50):
        assert policy.act(obs(dr=0, dc=0.5, right=1, up=1))[R0] in (DOWN, LEFT)
    seen = {policy.act(obs(dr=0, dc=0.5, right=1))[R0] for _ in range(50)}
    assert seen <= {UP, DOWN, LEFT}
    # It is really random, not a fixed fallback.
    assert len(seen) > 1


def test_greedy_stays_when_boxed_in_or_on_target():
    assert greedy().act(obs(dr=0.5, dc=0.5, up=1, down=1, left=1, right=1))[R0] == STAY
    assert greedy().act(obs(dr=0, dc=0))[R0] == STAY


def test_greedy_beats_random():
    greedy_mean = evaluate(POLICIES["greedy"], 3)["deliveries"]
    random_mean = evaluate(POLICIES["random"], 3)["deliveries"]
    assert greedy_mean >= 2 * random_mean, (greedy_mean, random_mean)


def test_evaluate_is_deterministic():
    assert evaluate(POLICIES["greedy"], 2) == evaluate(POLICIES["greedy"], 2)


def test_run_episode_stops_at_max_steps():
    env = MultiAgentWarehouseEnv(max_steps=10)
    result = run_episode(env, RandomPolicy(), seed=1)
    assert env.step_count == 10
    assert env.agents == []
    assert set(result) == {"deliveries", "collisions"}
