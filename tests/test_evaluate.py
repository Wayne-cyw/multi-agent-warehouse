"""Tests for the episode-level evaluation pieces."""

import pytest

from multi_agent_warehouse.agents import GreedyPolicy
from multi_agent_warehouse.evaluate import evaluate_episodes, evaluate_with_ci, summarise

# Pinned from evaluate_with_ci(GreedyPolicy, n_episodes=5) before evaluate.py was rebuilt.
PINNED = {
    "deliveries": (13.8, 4.356317711095002),
    "collisions": (312.8, 161.08099945058694),
}


def test_evaluate_with_ci_unchanged():
    stats = evaluate_with_ci(GreedyPolicy, n_episodes=5)

    for key, (mean, half) in PINNED.items():
        assert stats[key] == pytest.approx((mean, half), rel=1e-12)


def test_evaluate_episodes_returns_one_row_per_seed():
    rows = evaluate_episodes(GreedyPolicy, seeds=[3, 4, 5], n_robots=2)

    assert len(rows) == 3
    assert all(set(row) == {"deliveries", "collisions"} for row in rows)
    assert all(isinstance(v, int) for row in rows for v in row.values())


def test_evaluate_episodes_rejects_shaping_in_env_kwargs():
    with pytest.raises(ValueError):
        evaluate_episodes(GreedyPolicy, seeds=[0], env_kwargs={"shaping": True})


def test_shared_policy_matches_fresh_policies():
    shared = GreedyPolicy()

    assert evaluate_episodes(lambda: shared, [1, 2]) == evaluate_episodes(GreedyPolicy, [1, 2])


def test_summarise_single_episode_has_zero_half_width():
    stats = summarise([{"deliveries": 4, "collisions": 7}])

    assert stats == {"deliveries": (4.0, 0.0), "collisions": (7.0, 0.0)}


def test_evaluate_main_builds_rich_env_for_a_rich_model(tmp_path, monkeypatch, capsys):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("supersuit")
    from multi_agent_warehouse import agents, evaluate
    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(total_steps=512, n_envs=2, env_kwargs={"observation": "rich"})
    monkeypatch.setattr(agents, "MODEL_PATH", fit(cfg, tmp_path / "m"))

    evaluate.main(["--episodes", "1"])

    assert "rl" in capsys.readouterr().out
