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
