"""Tests for the optional PPO training stack (skipped when the `rl` extra is not installed)."""

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("supersuit")

from stable_baselines3 import PPO

from multi_agent_warehouse.train import train


def test_train_saves_a_loadable_model(tmp_path):
    path = train(total_steps=512, n_envs=2, out=str(tmp_path / "m"))

    assert path.exists()
    assert path.suffix == ".zip"
    assert isinstance(PPO.load(path), PPO)


def test_rl_policy_acts_for_all_robots(tmp_path):
    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv

    path = train(total_steps=512, n_envs=2, out=str(tmp_path / "m"))
    env = MultiAgentWarehouseEnv()
    observations, _ = env.reset(seed=0)
    policy = RLPolicy(path)
    policy.reset(0)

    actions = policy.act(observations)

    assert set(actions) == set(observations)
    assert all(a in range(5) for a in actions.values())


def test_available_policies_without_model(monkeypatch, tmp_path):
    from multi_agent_warehouse import agents

    monkeypatch.setattr(agents, "MODEL_PATH", tmp_path / "missing.zip")

    names = agents.available_policies()

    assert "rl" not in names
    assert {"random", "greedy"} <= set(names)


def test_available_policies_with_model(monkeypatch, tmp_path):
    from multi_agent_warehouse import agents

    path = train(total_steps=512, n_envs=2, out=str(tmp_path / "m"))
    monkeypatch.setattr(agents, "MODEL_PATH", path)

    assert "rl" in agents.available_policies()


def test_vec_env_exposes_truncation_flag_and_terminal_observation():
    import numpy as np
    import supersuit

    from multi_agent_warehouse.env import MultiAgentWarehouseEnv

    env = MultiAgentWarehouseEnv(shaping=True)
    vec_env = supersuit.pettingzoo_env_to_vec_env_v1(env)
    vec_env = supersuit.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    vec_env.reset()
    actions = np.zeros(vec_env.num_envs, dtype=np.int64)
    for _ in range(env.max_steps - 1):
        *_, infos = vec_env.step(actions)
        assert not any(info["TimeLimit.truncated"] for info in infos)
    _, _, dones, infos = vec_env.step(actions)

    assert all(dones)
    assert all(info["TimeLimit.truncated"] for info in infos)
    assert all("terminal_observation" in info for info in infos)
