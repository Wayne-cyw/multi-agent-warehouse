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

    with pytest.raises(ValueError, match="recurrent_ppo"):
        TrainConfig(algo="dqn")


def test_train_config_accepts_recurrent_ppo():
    from multi_agent_warehouse.train import TrainConfig

    assert TrainConfig(algo="recurrent_ppo").algo == "recurrent_ppo"


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

    with pytest.raises(ValueError, match="recurrent_ppo"):
        RLPolicy(tmp_path / "missing.zip", algo="dqn")


def test_rl_policy_rejects_wrong_observation_size():
    import numpy as np

    from multi_agent_warehouse.agents import MODEL_PATH, RLPolicy

    legacy = MODEL_PATH.with_name("ppo_v1_300k.zip")
    if not legacy.exists():
        pytest.skip("committed legacy model is missing")
    policy = RLPolicy(legacy)

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


def _recurrent_config(**kwargs):
    from multi_agent_warehouse.train import TrainConfig

    return TrainConfig(**{"algo": "recurrent_ppo", "n_envs": 2, "total_steps": 2048, **kwargs})


@pytest.fixture(scope="module")
def recurrent_model(tmp_path_factory):
    pytest.importorskip("sb3_contrib")
    from multi_agent_warehouse.train import fit

    return fit(_recurrent_config(), out=tmp_path_factory.mktemp("lstm") / "m")


def test_recurrent_fit_saves_a_loadable_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    import csv

    from sb3_contrib import RecurrentPPO

    from multi_agent_warehouse.train import fit

    cfg = _recurrent_config(gamma=0.95, checkpoint_steps=(1024,))

    path = fit(cfg, out=tmp_path / "final", log_dir=tmp_path)

    model = RecurrentPPO.load(path, device="cpu")
    assert isinstance(model, RecurrentPPO)
    assert model.num_timesteps >= 2048
    assert (model.n_steps, model.batch_size, model.gamma) == (128, 256, 0.95)
    assert len(list((tmp_path / "checkpoints").glob("step_*.zip"))) == 1
    with open(tmp_path / "progress.csv", newline="") as f:
        assert "train/value_loss" in csv.DictReader(f).fieldnames


def test_recurrent_same_seed_gives_identical_weights(tmp_path):
    pytest.importorskip("sb3_contrib")
    import torch
    from sb3_contrib import RecurrentPPO

    from multi_agent_warehouse.train import fit

    weights = []
    for name in ("a", "b"):
        path = fit(_recurrent_config(seed=7, total_steps=1024), out=tmp_path / name)
        weights.append(RecurrentPPO.load(path, device="cpu").policy.state_dict())

    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


def _lstm_state(policy):
    hidden, cell = policy._state
    return hidden.copy(), cell.copy()


def test_recurrent_policy_state_changes_between_steps(recurrent_model):
    import numpy as np

    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv

    env = MultiAgentWarehouseEnv()
    observations, _ = env.reset(seed=0)
    policy = RLPolicy(recurrent_model, algo="recurrent_ppo")
    policy.reset(0)

    actions = policy.act(observations)
    first = _lstm_state(policy)
    observations, *_ = env.step(actions)
    policy.act(observations)
    second = _lstm_state(policy)

    assert set(actions) == set(observations)
    assert first[0].shape == (1, 3, 256)
    assert not np.array_equal(first[0], second[0])


def test_recurrent_state_clears_on_reset(recurrent_model):
    import numpy as np

    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv

    env = MultiAgentWarehouseEnv()
    used = RLPolicy(recurrent_model, deterministic=True, algo="recurrent_ppo")
    observations, _ = env.reset(seed=0)
    used.reset(0)
    for _ in range(5):
        observations, *_ = env.step(used.act(observations))

    observations, _ = env.reset(seed=1)
    used.reset(0)
    fresh = RLPolicy(recurrent_model, deterministic=True, algo="recurrent_ppo")
    fresh.reset(0)

    assert used.act(observations) == fresh.act(observations)
    assert all(
        np.array_equal(a, b) for a, b in zip(_lstm_state(used), _lstm_state(fresh), strict=True)
    )


def test_recurrent_policy_handles_robot_count_change_after_reset(recurrent_model):
    from multi_agent_warehouse.agents import RLPolicy
    from multi_agent_warehouse.env import MultiAgentWarehouseEnv

    policy = RLPolicy(recurrent_model, algo="recurrent_ppo")
    for n_robots in (3, 4, 2):
        env = MultiAgentWarehouseEnv(n_robots=n_robots)
        observations, _ = env.reset(seed=n_robots)
        policy.reset(n_robots)
        for _ in range(3):
            actions = policy.act(observations)
            assert set(actions) == set(observations)
            observations, *_ = env.step(actions)
        assert _lstm_state(policy)[0].shape == (1, n_robots, 256)


def test_recurrent_policy_rejects_wrong_observation_size(recurrent_model):
    import numpy as np

    from multi_agent_warehouse.agents import RLPolicy

    policy = RLPolicy(recurrent_model, algo="recurrent_ppo")

    with pytest.raises(ValueError, match="expects 8"):
        policy.act({"robot_0": np.zeros(14, dtype=np.float32)})


def test_recurrent_ppo_without_sb3_contrib_names_the_rl_extra(monkeypatch):
    import sys

    from multi_agent_warehouse.train import model_class

    monkeypatch.setitem(sys.modules, "sb3_contrib", None)

    with pytest.raises(ImportError, match="rl"):
        model_class("recurrent_ppo")


def test_policy_env_kwargs_reads_sidecar(tmp_path):
    import json

    from multi_agent_warehouse.agents import policy_algo, policy_env_kwargs

    model = tmp_path / "m.zip"
    (tmp_path / "m.json").write_text(
        json.dumps({"algo": "recurrent_ppo", "env_kwargs": {"observation": "rich"}})
    )

    assert policy_env_kwargs(model) == {"observation": "rich"}
    assert policy_algo(model) == "recurrent_ppo"


def test_policy_env_kwargs_defaults_without_sidecar(tmp_path):
    from multi_agent_warehouse.agents import policy_algo, policy_env_kwargs

    assert policy_env_kwargs(tmp_path / "none.zip") == {}
    assert policy_algo(tmp_path / "none.zip") == "ppo"


def test_fit_writes_a_sidecar_next_to_the_model(tmp_path):
    from multi_agent_warehouse.agents import policy_env_kwargs
    from multi_agent_warehouse.train import TrainConfig, fit

    cfg = TrainConfig(total_steps=512, n_envs=2, env_kwargs={"observation": "rich"})
    path = fit(cfg, tmp_path / "m")

    assert policy_env_kwargs(path) == {"observation": "rich"}


def test_rl_policy_needs_sb3_contrib_for_a_recurrent_model(monkeypatch, tmp_path):
    import importlib.util
    import json

    from multi_agent_warehouse import agents

    model = tmp_path / "m.zip"
    model.write_bytes(b"x")
    (tmp_path / "m.json").write_text(json.dumps({"algo": "recurrent_ppo"}))
    monkeypatch.setattr(agents, "MODEL_PATH", model)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "sb3_contrib" else real(name, *a),
    )

    assert "rl" not in agents.available_policies()

    (tmp_path / "m.json").write_text(json.dumps({"algo": "ppo"}))

    assert "rl" in agents.available_policies()


def test_train_cli_default_out_is_not_the_shipped_model(monkeypatch):
    from multi_agent_warehouse import train as train_module

    saved = []
    monkeypatch.setattr(train_module, "fit", lambda cfg, out, **kw: saved.append((cfg, out)))

    train_module.main(["--steps", "1", "--seed", "0", "--no-bootstrap-truncation"])

    cfg, out = saved[0]
    assert str(out) == "models/ppo_retrained"
    assert str(out) != "models/ppo_multi_agent_warehouse"
    assert cfg.bootstrap_truncation is False


def test_fit_sidecar_follows_a_dotted_model_name(tmp_path):
    from multi_agent_warehouse.train import TrainConfig, fit

    path = fit(TrainConfig(total_steps=512, n_envs=2), tmp_path / "ppo_v1.5k")

    assert path == tmp_path / "ppo_v1.5k"
    assert path.exists()
    assert (tmp_path / "ppo_v1.json").exists()
