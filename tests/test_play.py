"""Smoke tests for the live demo: run `play.py` headless for a few frames."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_play(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `play.py` from the repo root with a dummy video driver."""
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    return subprocess.run(
        [sys.executable, "play.py", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_play_headless_smoke(policy):
    result = run_play("--policy", policy, "--frames", "30", "--fps", "1000")
    assert result.returncode == 0, result.stderr


def test_play_unknown_policy_is_rejected():
    result = run_play("--policy", "bogus", "--frames", "5")
    assert result.returncode != 0


def test_play_rejects_too_many_robots():
    result = run_play("--robots", "99", "--frames", "5")
    assert result.returncode == 2
