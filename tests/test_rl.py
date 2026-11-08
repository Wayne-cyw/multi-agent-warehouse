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


def test_train_config_roundtrips_through_json():
    import json

    from multi_agent_warehouse.train import TrainConfig

    cfg = TrainConfig(
        net_arch=(32, 16),
        robot_counts=(2, 4),
        checkpoint_steps=(100, 200),
        env_kwargs={"shaping_scale": 0.5, "collision_penalty": -1.0},
    )

    assert TrainConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_train_config_rejects_unknown_algo():
    from multi_agent_warehouse.train import TrainConfig

    with pytest.raises(ValueError):
        TrainConfig(algo="dqn")


def test_fit_writes_checkpoints_progress_and_config(tmp_path):
    import csv
    import json

    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(n_envs=2, total_steps=3072, checkpoint_steps=(1024, 2048))

    path = fit(cfg, out=tmp_path / "final", log_dir=tmp_path)

    assert path == tmp_path / "final.zip"
    assert path.exists()
    checkpoints = sorted((tmp_path / "checkpoints").glob("step_*.zip"))
    assert len(checkpoints) == 2
    steps = [int(p.stem.removeprefix("step_")) for p in checkpoints]
    assert steps[0] >= 1024 and steps[1] >= 2048
    with open(tmp_path / "progress.csv", newline="") as f:
        columns = set(csv.DictReader(f).fieldnames)
    assert {
        "time/total_timesteps",
        "rollout/ep_rew_mean",
        "train/entropy_loss",
        "train/value_loss",
        "train/explained_variance",
    } <= columns
    saved = json.loads((tmp_path / "config.json").read_text())
    assert TrainConfig.from_dict(saved) == cfg


def _fit_state_dict(tmp_path, name, seed):
    from multi_agent_warehouse.train import TrainConfig, fit

    path = fit(TrainConfig(n_envs=2, total_steps=1024, seed=seed), out=tmp_path / name)
    return PPO.load(path, device="cpu").policy.state_dict()


def test_same_seed_gives_identical_weights(tmp_path):
    import torch

    first = _fit_state_dict(tmp_path, "a", 7)
    second = _fit_state_dict(tmp_path, "b", 7)

    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_different_seeds_give_different_weights(tmp_path):
    import torch

    first = _fit_state_dict(tmp_path, "a", 7)
    second = _fit_state_dict(tmp_path, "b", 8)

    assert not all(torch.equal(first[k], second[k]) for k in first)


def test_hyperparameters_reach_the_model(tmp_path):
    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(
        n_envs=1, total_steps=256, batch_size=128, ent_coef=0.01, gamma=0.95, net_arch=(32, 32)
    )

    model = PPO.load(fit(cfg, out=tmp_path / "m"), device="cpu")

    assert model.ent_coef == 0.01
    assert model.gamma == 0.95
    assert model.policy.mlp_extractor.policy_net[0].out_features == 32


def test_mixed_robot_counts_give_one_sub_env_per_robot():
    from multi_agent_warehouse.train import TrainConfig, _make_vec_env

    vec_env = _make_vec_env(TrainConfig(n_envs=3, robot_counts=(2, 3, 4)))

    assert vec_env.num_envs == 9
    assert vec_env.reset().shape[0] == 9
    vec_env.close()


def _infos_at_time_limit(bootstrap_truncation):
    import numpy as np

    from multi_agent_warehouse.config import DEFAULT_MAX_STEPS
    from multi_agent_warehouse.train import TrainConfig, _make_vec_env

    vec_env = _make_vec_env(TrainConfig(n_envs=1, bootstrap_truncation=bootstrap_truncation))
    vec_env.reset()
    actions = np.zeros(vec_env.num_envs, dtype=np.int64)
    for _ in range(DEFAULT_MAX_STEPS):
        _, _, dones, infos = vec_env.step(actions)
    vec_env.close()
    assert all(dones)
    return infos


def test_bootstrap_keeps_the_flag():
    infos = _infos_at_time_limit(True)

    assert all(info["TimeLimit.truncated"] for info in infos)
    assert all("terminal_observation" in info for info in infos)


def test_disabling_bootstrap_removes_the_flag():
    infos = _infos_at_time_limit(False)

    assert all("TimeLimit.truncated" not in info for info in infos)
    assert all("terminal_observation" in info for info in infos)


def test_main_draws_a_random_seed_when_none_is_given(monkeypatch, tmp_path, capsys):
    from multi_agent_warehouse import train as train_module

    seeds = []

    def fake_fit(cfg, out, log_dir=None, verbose=0):
        seeds.append(cfg.seed)
        return out.with_suffix(".zip")

    monkeypatch.setattr(train_module, "fit", fake_fit)
    monkeypatch.chdir(tmp_path)

    # Three draws from 1000 seeds all colliding has probability 1e-6.
    for _ in range(3):
        train_module.main([])

    printed = capsys.readouterr().out
    assert len(set(seeds)) > 1
    assert all(0 <= seed < 1000 for seed in seeds)
    assert all(str(seed) in printed for seed in seeds)


def test_deterministic_rl_policy_repeats_actions(tmp_path):
    import numpy as np

    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv
    from multi_agent_warehouse.train import TrainConfig, fit

    path = fit(TrainConfig(n_envs=2, total_steps=512), out=tmp_path / "m")
    observations, _ = MultiAgentWarehouseEnv().reset(seed=0)
    policy = RLPolicy(path, deterministic=True)

    first = policy.act(observations)
    second = policy.act(observations)

    assert first == second
    assert np.all([a in range(5) for a in first.values()])


def test_rl_policy_rejects_unknown_algo(tmp_path):
    from multi_agent_warehouse.agents import RLPolicy

    with pytest.raises(ValueError):
        RLPolicy(tmp_path / "missing.zip", algo="dqn")


def test_rl_policy_rejects_wrong_observation_size():
    import numpy as np

    from multi_agent_warehouse.agents import MODEL_PATH, RLPolicy

    if not MODEL_PATH.exists():
        pytest.skip("committed model is missing")
    policy = RLPolicy(MODEL_PATH)

    with pytest.raises(ValueError, match="expects 8"):
        policy.act({"robot_0": np.zeros(14, dtype=np.float32)})


def test_rich_observation_model_trains_and_acts(tmp_path):
    import numpy as np

    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv
    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(n_envs=2, total_steps=3072, env_kwargs={"observation": "rich"})
    path = fit(cfg, out=tmp_path / "rich")
    policy = RLPolicy(path)

    rich_obs, _ = MultiAgentWarehouseEnv(observation="rich").reset(seed=0)
    actions = policy.act(rich_obs)
    assert set(actions) == set(rich_obs)
    assert all(a in range(5) for a in actions.values())

    basic_obs, _ = MultiAgentWarehouseEnv().reset(seed=0)
    with pytest.raises(ValueError, match="expects 14"):
        policy.act(basic_obs)
    assert np.all([o.shape == (14,) for o in rich_obs.values()])
