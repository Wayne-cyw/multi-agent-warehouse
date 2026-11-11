# 13. Testing and tooling

[Index](README.md) · Previous: [12 evaluation](12-evaluation.md) · Next: [14 generalization](14-generalization.md)

## Test-driven development (TDD)

For each task the plan has you **write failing tests first**, run them to see them fail, then implement
until they pass. Seeing a test fail first proves it can fail; otherwise a test that never tests
anything looks identical to a good one.

## Test files

Tests live flat in `tests/`, one file per module, named `test_<module>.py`, with no `__init__.py`.
Without `__init__.py`, pytest's default import mode needs unique basenames.

| File | Tests | What it protects | State |
| --- | --- | --- | --- |
| `test_rules.py` | 11 | The move-resolution cases ([09](09-move-resolution.md)) and "inputs are not mutated" | Exists |
| `test_config.py` | 4 | Layout is 8 x 8, only `.` and `#`, enough free cells, **every free cell reachable** (BFS) | Exists |
| `test_env.py` | 19 | API compliance, spaces, seeding, rewards, observations, truncation, invalid input | Exists |
| `test_render.py` | 11 | Frame shape and exact pixels, `None` mode, double `close()`, rendering leaves the trajectory and RNG untouched, pygame is not imported until the first render | Exists |
| `test_agents.py` | 10 | Policy rules, determinism, "greedy >= 2x random", evaluation means and confidence intervals | Exists |
| `test_play.py` | 3 | Headless demo smoke tests via `subprocess` | Exists |
| `test_rl.py` | 4 | Training saves a loadable model; `RLPolicy` acts for every robot; `available_policies()` with and without the model; skipped if SB3 is absent | Exists |

As of 2026-11-03, `SDL_VIDEODRIVER=dummy .venv/bin/pytest -q` reports **59 passed** *(verified)*; **63 passed** after Task 8 (2026-11-07), and `ruff check` and `ruff format --check` are clean.

### Tests derive sizes from `LAYOUT`

The plan says tests compute sizes from the map (for example `LAST = len(LAYOUT) - 1`) instead of
writing `7`. So editing the map does not silently break unrelated tests, and the one test that
*should* fail when the size changes (`test_layout_is_8_by_8`) says so clearly.

### Edge cases (the plan's "Review Focus")

Each is pinned by a test:

1. `step()` after the episode ended returns five empty dicts.
2. Missing, extra or out-of-range actions raise `ValueError`.
3. `n_robots` of 0 or 7, or an unknown `render_mode`, raises `ValueError`.
4. `render()` with `render_mode=None` returns `None`; `close()` twice is safe; rendering never changes
   the trajectory.
5. `play.py` works with no RL model file.

### The PettingZoo API test

`test_parallel_api` runs `pettingzoo.test.parallel_api_test(MultiAgentWarehouseEnv(), num_cycles=300)`.
If it complains, **fix the env, not the test** ([06](06-pettingzoo-and-gymnasium.md)).

## Running things

Always use the project's virtual environment:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

CI-equivalent check (used at the end of Stage 4):

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && SDL_VIDEODRIVER=dummy .venv/bin/pytest -q
```

## Lint and format

Ruff, exactly as CI runs it: rules `E, F, I, B, UP` (errors, unused code, import order, likely bugs,
modern-syntax upgrades), line length 100, and `ruff format --check`. "Passes locally" should mean
"passes CI".

## Headless Pygame

Any test file that imports `pygame` or the renderer sets, at the top:

```python
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
```

`dummy` tells SDL (Pygame's backend) to render without a display, so tests pass on machines and CI
with no screen. `setdefault` lets a developer override it.

## Code conventions

Type hints on public functions; a docstring on every module and class; all randomness from
`np.random.default_rng` and no global RNG; no pandas or SciPy.

## Check yourself

1. Why write the test before the implementation?
2. Why must test basenames be unique here?
3. What does `test_all_free_cells_are_reachable_from_each_other` stop you from doing by accident?
4. What does `SDL_VIDEODRIVER=dummy` do, and why `setdefault`?
5. If `parallel_api_test` fails, what do you change, the test or the env?
