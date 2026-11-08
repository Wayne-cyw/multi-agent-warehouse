# Multi-Agent Warehouse Implementation Plan

> **Status 2026-11-03:** Tasks 1 to 6 (Stages 1 to 4) are done and merged, so their steps are ticked.
> Task 7 (Stage 5, RL) is done on `feat/ppo-training-script` and Task 8 on `feat/ppo-trained-agent` (2026-11-07). Where the text below differs from the repo, the repo
> wins: there is no `warehouse_rl/` in this repository, the README is at the repo root (not
> `multi_agent_warehouse/README.md`), work used one branch and PR per task instead of a single
> `multi-agent-warehouse` branch, and `evaluate` now defaults to 100 episodes with 95% confidence
> intervals (`evaluate_with_ci`), so the 20-episode wording in Task 4 is historical.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small PettingZoo + Pygame multi-robot warehouse with two scripted policies, a live demo and a tiny evaluation, then (stretch) a PPO agent whose saved weights the demo can load.

**Architecture:** A pure `resolve_moves` function is the physics core, wrapped by `MultiAgentWarehouseEnv` (a `ParallelEnv`). A `Renderer` always draws offscreen and returns an RGB array; `human` mode also blits to a window. Policies share a two-method interface (`reset`, `act`) so the demo, the evaluation and the RL agent all plug into the same loop.

**Tech Stack:** Python 3.10+ (venv `.venv`, Python 3.11), NumPy, Gymnasium, PettingZoo, Pygame, pytest, ruff. Stage 5 only: stable-baselines3, supersuit, torch. Pillow (already installed) for the README GIF.

**Spec:** `docs/superpowers/specs/2026-10-28-multi-agent-warehouse-design.md`. Read it first: it fixes the rules, the observation, the rewards and the rendering numbers that the tests below assert.

## Global Constraints

- Always run tools through the venv: `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python`.
- Work on branch `multi-agent-warehouse` (created in Task 1). Commit messages `type(scope): message`. Do not modify anything under `warehouse_rl/` or its tests.
- Ruff as CI runs it: rules `E,F,I,B,UP`, line length 100, and `ruff format --check`.
- Type hints on public functions; a docstring on every module and class.
- All randomness from `np.random.default_rng`; no global RNG. No pandas, no SciPy.
- Actions: 0 stay, 1 up (row - 1), 2 down (row + 1), 3 left (col - 1), 4 right (col + 1).
- Observation: `Box(-1, 1, (8,), float32)`: `[row, col, dr, dc, up_blocked, down_blocked, left_blocked, right_blocked]`; the first four divided by `max(H, W) - 1`.
- Rewards: delivery +1.0; every step -0.01 (all robots); collision -0.1 (each collided robot); shaping +0.02 x (Manhattan distance to the held target before the step - after), off by default.
- Defaults: 8 x 8 layout from the spec, `n_robots=3` (allowed 1 to 6), `max_steps=200`, `render_fps=10`.
- Settings live in one file, `multi_agent_warehouse/config.py`: `LAYOUT`, `DEFAULT_N_ROBOTS`, `MAX_ROBOTS`, `DEFAULT_MAX_STEPS`, `DELIVERY_REWARD`, `STEP_PENALTY`, `COLLISION_PENALTY`, `SHAPING_SCALE`, `RENDER_MODES`, `RENDER_FPS`. Every other module imports them from there and never re-defines a number or the map. Tests derive sizes from `LAYOUT` (for example `len(LAYOUT) - 1`) instead of writing `7`.
- Tests live flat in `tests/` as `test_<module>.py` (no `__init__.py`, so basenames must be unique). Headless tests need `SDL_VIDEODRIVER=dummy`; set it with `os.environ.setdefault` at the top of any test file that imports `pygame` or the renderer.
- Policies use only observations. No policy reads env internals.

## Review Focus

Inputs the spec implies but the main tests would not otherwise hit, most likely first. Each is pinned in the task that owns the code.

1. `step()` called after the episode ended (`env.agents == []`): returns five empty dicts instead of raising. (Task 2)
2. `step()` with missing, extra or out-of-range actions: raises `ValueError` naming the problem. (Task 2)
3. `MultiAgentWarehouseEnv(n_robots=0)` or `n_robots=7`, or an unknown `render_mode`: raises `ValueError`. (Task 2)
4. `render()` with `render_mode=None` returns `None` and opens no window; `close()` twice is safe; rendering does not change the trajectory. (Task 3)
5. `play.py` without the RL model file, and with `--policy rl` but no file, still runs (falls back with a one-line message). (Task 5 and Task 8)

---

# Stage 1: Rules and environment

### Task 1: Move resolution

**Files:**
- Create: `multi_agent_warehouse/__init__.py`, `multi_agent_warehouse/rules.py`
- Modify: `pyproject.toml` (packages include `multi_agent_warehouse*`)
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces in `multi_agent_warehouse/rules.py`: `Cell = tuple[int, int]`; constants `STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4`; `ACTION_DELTAS: tuple[Cell, ...]` indexed by action; `resolve_moves(positions: dict[str, Cell], actions: dict[str, int], shelves: np.ndarray) -> tuple[dict[str, Cell], set[str]]` returning the new positions of every robot and the set of collided robot ids. Inputs are not mutated.

- [x] **Step 1: Create branch and package skeleton.** `git switch -c multi-agent-warehouse`; create `multi_agent_warehouse/__init__.py` (docstring only for now); in `pyproject.toml` change `include = ["warehouse_rl*"]` to `include = ["warehouse_rl*", "multi_agent_warehouse*"]`. Run `.venv/bin/pip install -e .` once so the package is importable.

- [x] **Step 2: Write the failing tests** in `tests/test_rules.py`, using a 5 x 5 grid with no shelves unless stated and a helper `resolve(pos, acts, shelves=None)` that calls `resolve_moves`. Each test below is one function; coordinates are `(row, col)`:

| Test | Positions and actions | Asserts |
| --- | --- | --- |
| `test_free_move` | a (2,2) RIGHT | a at (2,3); collided empty |
| `test_wall_and_shelf_bump_are_not_collisions` | a (0,0) UP; and with shelf at (2,3): a (2,2) RIGHT | a unmoved; collided empty (both cases) |
| `test_two_into_one` | a (2,1) RIGHT, b (2,3) LEFT | both unmoved; collided == {a, b} |
| `test_three_into_one` | a (2,1) RIGHT, b (2,3) LEFT, c (1,2) DOWN | all unmoved; collided == {a, b, c} |
| `test_into_a_stayer` | a (2,1) RIGHT, b (2,2) STAY | a unmoved; collided == {a} |
| `test_swap` | a (2,1) RIGHT, b (2,2) LEFT | both unmoved; collided == {a, b} |
| `test_follow_the_leader` | a (2,1) RIGHT, b (2,2) RIGHT | a (2,2), b (2,3); collided empty |
| `test_cascade` | a (2,0) RIGHT, b (2,1) RIGHT, c (2,3) LEFT | all unmoved; collided == {a, b, c} |
| `test_behind_a_wall_bump` | shelf (2,3); a (2,1) RIGHT, b (2,2) RIGHT | a unmoved, b unmoved; collided == {a} |
| `test_rotation_is_allowed` | a (1,1) RIGHT, b (1,2) DOWN, c (2,2) LEFT, d (2,1) UP | a (1,2), b (2,2), c (2,1), d (1,1); collided empty |
| `test_inputs_not_mutated` | any case | the input dicts equal copies taken before the call |

- [x] **Step 3: Run to verify failure.** `.venv/bin/pytest tests/test_rules.py -q` fails with an import error.

- [x] **Step 4: Implement `resolve_moves`.** Algorithm: proposals are `pos + delta`, replaced by `pos` when out of bounds or on a shelf. Then repeat until nothing changes: a moving robot (proposal != pos) is blocked if another robot's proposal equals its proposal, or if another robot's proposal is this robot's cell while this robot's proposal is that robot's cell (swap); a blocked robot's proposal becomes its own cell. Collided = robots that were moving at the start and ended up blocked. Iterate robots in sorted-id order for determinism.

- [x] **Step 5: Run to verify pass.** `.venv/bin/pytest tests/test_rules.py -q` passes (11 tests).

- [x] **Step 6: Commit.** `git add pyproject.toml multi_agent_warehouse tests/test_rules.py && git commit -m "feat(warehouse): add move resolution"`

### Task 2: The environment

**Files:**
- Create: `multi_agent_warehouse/config.py`, `multi_agent_warehouse/env.py`
- Modify: `multi_agent_warehouse/__init__.py` (export `MultiAgentWarehouseEnv`)
- Test: `tests/test_config.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: `resolve_moves`, `ACTION_DELTAS`, `Cell` from Task 1.
- Produces in `multi_agent_warehouse/config.py` (plain constants, no logic): `LAYOUT: tuple[str, ...]` (the spec's 8 rows of 8 characters, `.` free and `#` shelf); `DEFAULT_N_ROBOTS = 3`, `MAX_ROBOTS = 6`, `DEFAULT_MAX_STEPS = 200`; `DELIVERY_REWARD = 1.0`, `STEP_PENALTY = -0.01`, `COLLISION_PENALTY = -0.1`, `SHAPING_SCALE = 0.02`; `RENDER_MODES = ("human", "rgb_array")`, `RENDER_FPS = 10`.
- Produces in `multi_agent_warehouse/env.py`: `class MultiAgentWarehouseEnv(ParallelEnv)` with `__init__(self, n_robots: int = DEFAULT_N_ROBOTS, max_steps: int = DEFAULT_MAX_STEPS, shaping: bool = False, render_mode: str | None = None)`; `metadata = {"name": "multi_agent_warehouse_v0", "render_modes": ["human", "rgb_array"], "render_fps": 10}` (built from the config constants); the standard `reset(seed=None, options=None)`, `step(actions)`, `observation_space(agent)`, `action_space(agent)`, `render()`, `close()`; public state `positions`, `targets`, `shelves` (bool `(8, 8)` array built from `config.LAYOUT`), `step_count`, `total_deliveries`, `total_collisions`, `hud_note: str = ""`. `env.py` defines no settings of its own. `render()` and `close()` are stubs here (Task 3 fills them in).

- [x] **Step 1: Write the failing tests.** First `tests/test_config.py`: `test_layout_is_8_by_8` (8 rows, each 8 characters); `test_layout_uses_only_floor_and_shelf_characters` (only `.` and `#`); `test_layout_has_room_for_every_robot_and_a_different_target` (more free cells than `MAX_ROBOTS`); `test_all_free_cells_are_reachable_from_each_other` (a breadth-first search from any free cell reaches them all, so an edited map cannot seal off a target). Then `tests/test_env.py`, where `LAST = len(LAYOUT) - 1` (7) and `MID = LAST // 2`, the parked-robot helper uses the corners and the middle row, and the whole of rows 0, 3 and `MID` plus the four corners must be free:
  - `test_parallel_api`: `parallel_api_test(MultiAgentWarehouseEnv(), num_cycles=300)` from `pettingzoo.test`.
  - `test_spaces`: `observation_space("robot_0")` is a `Box` with shape `(8,)`, dtype `float32`, bounds -1 and 1; `action_space` is `Discrete(5)`.
  - `test_reset_places_robots_on_distinct_free_cells_with_valid_targets`: for 20 seeds, positions are distinct and not shelves; each target is free and differs from its robot's cell.
  - `test_same_seed_same_trajectory`: two envs reset with seed 7 and fed identical action sequences (from a separate `default_rng(0)`) give equal observations and rewards for 50 steps; a different seed differs.
  - `test_delivery`: set `env.positions["robot_0"] = (0, 0)` and `env.targets["robot_0"] = (0, 1)`, others far away; step RIGHT for robot_0. Reward is `1.0 - 0.01`; `infos["robot_0"]["deliveries"] == 1`; the new target is free and differs from `(0, 1)`; `env.total_deliveries == 1`.
  - `test_collision_reward`: place robot_0 at (0, 0) and robot_1 at (0, 2), both moving into (0, 1). Each reward is `-0.01 - 0.1`; `infos[...]["collided"]` is True for both; `env.total_collisions == 2`.
  - `test_bump_has_no_penalty`: robot_0 at (0, 0) moves UP: reward `-0.01`, `collided` False.
  - `test_observation_values`: robot at (0, 0) with target `(LAST, LAST)`: obs equals `[0, 0, 1, 1, 1, 0, 1, 0]` (up and left blocked by the edge). The delivery test likewise expects offsets divided by `LAST`.
  - `test_neighbour_flag_for_other_robot`: a robot directly right of robot_0 sets robot_0's right flag to 1.
  - `test_shaping`: with `shaping=True`, robot at (0, 0), target (0, 3), moving RIGHT gives `-0.01 + 0.02`; with the default it gives `-0.01`.
  - `test_truncates_at_max_steps`: `max_steps=5`: the fifth `step` returns truncations all True and `env.agents == []`; earlier steps return all False.
  - **Review Focus 1:** `test_step_after_end_returns_empty`: after truncation, `step({})` returns five empty dicts and does not raise.
  - **Review Focus 2:** `test_invalid_actions_raise`: a missing agent key, an extra key, an action of 5 or -1, and a non-integer each raise `ValueError`.
  - **Review Focus 3:** `test_invalid_constructor_args_raise`: `n_robots=0`, `n_robots=7`, `render_mode="bogus"` each raise `ValueError`.

- [x] **Step 2: Run to verify failure.** `.venv/bin/pytest tests/test_env.py -q` fails with an import error.

- [x] **Step 3: Implement `config.py` and `MultiAgentWarehouseEnv`.** Write `config.py` first (constants only). In `env.py` import every setting from it; do not define any. Follow the spec's step order. Build `shelves` from `config.LAYOUT` once and read the grid height and width from it (nothing else in the code may assume the grid size). `reset` creates or continues one `np.random.default_rng` (a new generator when `seed` is given), samples distinct start cells and targets from the free cells, zeroes counters, sets `agents = possible_agents[:]`, and returns observations and infos. Target resampling after a delivery draws a free cell different from the robot's current cell. Cache spaces with `functools.lru_cache` per the PettingZoo convention. Validation raises `ValueError` before any state changes.

- [x] **Step 4: Run to verify pass.** `.venv/bin/pytest tests/test_env.py -q` passes. If `parallel_api_test` complains about something, fix the env, not the test.

- [x] **Step 5: Commit.** `git add multi_agent_warehouse tests/test_config.py tests/test_env.py && git commit -m "feat(warehouse): add ParallelEnv with rewards and observations"`

---

# Stage 2: Renderer

### Task 3: Pygame renderer and `env.render`

**Files:**
- Create: `multi_agent_warehouse/render.py`
- Modify: `multi_agent_warehouse/env.py` (`render`, `close`, lazy renderer)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: the public state of `MultiAgentWarehouseEnv` from Task 2.
- Produces in `multi_agent_warehouse/render.py`: constants `CELL = 64`, `HUD_H = 40`, `TARGET_INSET = 8`, `TARGET_WIDTH = 6` (drawing numbers stay in `render.py`; only game settings live in `config.py`), `FLOOR_COLOR`, `SHELF_COLOR`, `ROBOT_COLORS` (six distinct RGB tuples); `class Renderer` with `__init__(self, shelves: np.ndarray, window: bool = False, fps: int = 10)`, `cell_origin(self, row: int, col: int) -> tuple[int, int]` (top-left pixel `(x, y)` including the HUD offset), `draw(self, positions: dict[str, Cell], targets: dict[str, Cell], hud: list[str]) -> np.ndarray` returning `(HUD_H + H * CELL, W * CELL, 3)` uint8, and `close(self) -> None` (safe to call twice). `env.render()` returns the array in `rgb_array` mode, shows the window and returns `None` in `human` mode, and returns `None` when `render_mode is None`. The HUD lines are step, deliveries, collisions and `env.hud_note`.

- [x] **Step 1: Write the failing tests** in `tests/test_render.py`, with `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` as the first line after the docstring:
  - `test_draw_shape_and_dtype`: a shelf array built from `config.LAYOUT` (8 x 8) gives shape `(HUD_H + 8 * CELL, 8 * CELL, 3)`, which is `(552, 512, 3)`, and dtype uint8.
  - `test_pixels`: robot_0 at (0, 0) and its target at (0, 3), robot_1 at (0, 1): the centre pixel of cell (0, 0) equals `ROBOT_COLORS[0]`; the centre of cell (1, 1) (a shelf in `LAYOUT`) equals `SHELF_COLOR`; the pixel at `cell_origin(0, 3) + (TARGET_INSET + 2, CELL // 2)` equals `ROBOT_COLORS[0]`; a pixel at `cell_origin(4, 4) + (2, 2)` (empty floor) equals `FLOOR_COLOR`.
  - `test_env_rgb_array_render`: `MultiAgentWarehouseEnv(render_mode="rgb_array")` returns an array of the same shape after `reset()`.
  - **Review Focus 4:** `test_render_mode_none_returns_none`; `test_close_twice_is_safe`; `test_rendering_does_not_change_trajectory`: two envs with the same seed and actions, one rendering every step, give identical observations and rewards.
  - `test_hud_note_changes_the_frame`: setting `env.hud_note = "PAUSED"` changes at least one pixel in the HUD band (compare arrays before and after).

- [x] **Step 2: Run to verify failure.** `.venv/bin/pytest tests/test_render.py -q` fails with an import error.

- [x] **Step 3: Implement `Renderer` and wire `render()` and `close()` in the env.** Always draw on an offscreen `pygame.Surface`; convert with `pygame.surfarray.array3d` and transpose to `(rows, cols, 3)`. Draw order: floor, shelves, targets (outlined squares), robots (filled circles), HUD text (`pygame.font.Font(None, 24)`). In window mode use `pygame.display.set_mode`, blit, `pygame.event.pump()`, and a `pygame.time.Clock` tick at `fps`. Import pygame inside `render.py` only, and create the renderer lazily on first `render()` so `MultiAgentWarehouseEnv` without a render mode never imports pygame.

- [x] **Step 4: Run to verify pass.** `.venv/bin/pytest tests/test_render.py tests/test_env.py -q` passes.

- [x] **Step 5: Commit.** `git add multi_agent_warehouse tests/test_render.py && git commit -m "feat(warehouse): add pygame renderer"`

---

# Stage 3: Policies, evaluation and demo

### Task 4: Policies and evaluation

**Files:**
- Create: `multi_agent_warehouse/agents.py`, `multi_agent_warehouse/evaluate.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `MultiAgentWarehouseEnv`, `STAY, UP, DOWN, LEFT, RIGHT`.
- Produces in `agents.py`: `class Policy(Protocol)` with `reset(self, seed: int | None = None) -> None` and `act(self, observations: dict[str, np.ndarray]) -> dict[str, int]`; `RandomPolicy`; `GreedyPolicy` (both constructed with no arguments); `POLICIES: dict[str, Callable[[], Policy]]` with keys `"random"` and `"greedy"`.
- Produces in `evaluate.py`: `run_episode(env: MultiAgentWarehouseEnv, policy: Policy, seed: int) -> dict[str, int]` returning `{"deliveries": ..., "collisions": ...}` (the env's totals at the end); `evaluate(make_policy: Callable[[], Policy], n_episodes: int = 20, first_seed: int = 1000, n_robots: int = DEFAULT_N_ROBOTS) -> dict[str, float]` returning `{"deliveries": mean, "collisions": mean}` per episode, always with shaping off; `main() -> None` printing a table of every available policy; `if __name__ == "__main__": main()`.

- [x] **Step 1: Write the failing tests** in `tests/test_agents.py`. Observation helper `obs(dr, dc, up=0, down=0, left=0, right=0)` returns the 8-vector with zeros for position:
  - `test_random_policy_valid_and_seeded`: actions are ints in 0..4, one per observed agent; two policies reset with seed 3 give the same actions.
  - `test_greedy_moves_along_larger_axis`: target to the right (`dc=0.5, dr=0.1`) gives RIGHT; target below (`dr=0.5, dc=0.1`) gives DOWN; a tie gives DOWN or UP (rows win).
  - `test_greedy_tries_other_axis_when_blocked`: `dc=0.5, dr=0.1, right=1` gives DOWN.
  - `test_greedy_sidesteps_when_other_axis_is_zero`: `dr=0, dc=0.5, right=1, up=1` gives DOWN or LEFT (the only unblocked directions), over 50 calls; with `right=1` only, the action is never RIGHT or STAY.
  - `test_greedy_stays_when_boxed_in_or_on_target`: all four flags 1 gives STAY; `dr=dc=0` gives STAY.
  - `test_greedy_beats_random`: over seeds 1000 to 1009 (3 robots), `evaluate(POLICIES["greedy"], 10)["deliveries"]` is at least twice `evaluate(POLICIES["random"], 10)["deliveries"]`. If this fails, print both means and fix the greedy rule (not the threshold) unless the means show the threshold itself is unreasonable.
  - `test_evaluate_is_deterministic`: two identical calls return equal dicts.
  - `test_run_episode_stops_at_max_steps`: an env with `max_steps=10` ends after 10 steps.

- [x] **Step 2: Run to verify failure.** `.venv/bin/pytest tests/test_agents.py -q` fails with an import error.

- [x] **Step 3: Implement `agents.py`.** `RandomPolicy` and `GreedyPolicy` each own `np.random.default_rng(seed)` created in `reset`, and iterate robot ids in sorted order. The greedy rule is exactly the one in the spec.

- [x] **Step 4: Implement `evaluate.py`.** `run_episode` resets the env and the policy with the same seed, then loops until `env.agents` is empty, passing observations straight to `policy.act`. `evaluate` builds a fresh policy and a fresh env per episode. `main` prints a fixed-width table, columns `policy`, `deliveries/episode`, `collisions/episode`, over `POLICIES` (and the `rl` entry when it exists, added in Task 8).

- [x] **Step 5: Run to verify pass.** `.venv/bin/pytest tests/test_agents.py -q` passes. Then `.venv/bin/python -m multi_agent_warehouse.evaluate` prints a two-row table; copy it for the README (Task 6).

- [x] **Step 6: Commit.** `git add multi_agent_warehouse tests/test_agents.py && git commit -m "feat(warehouse): add random and greedy policies and evaluation"`

### Task 5: The demo

**Files:**
- Create: `play.py` (repo root)
- Test: `tests/test_play.py`

**Interfaces:**
- Consumes: `MultiAgentWarehouseEnv`, `POLICIES`, `Policy`.
- Produces: `play.py` with `main(argv: list[str] | None = None) -> int`. Flags: `--policy {random,greedy,rl}` (default `greedy`), `--robots` (default `DEFAULT_N_ROBOTS`), `--seed` (default 0), `--fps` (default 10), `--frames` (default 0 = run until quit). Keys: space pause, R reset (same seed + 1), P cycle policy, Esc or Q quit. `hud_note` shows e.g. `greedy | PAUSED`. Available modes come from `POLICIES` plus `rl` only when its model file exists (Task 8 adds that; for now `rl` is never available). Requesting an unavailable `--policy` prints one line naming it and falls back to `greedy`.

- [x] **Step 1: Write the failing tests** in `tests/test_play.py`, running `play.py` through `subprocess.run([sys.executable, "play.py", "--frames", "30"], env={..., "SDL_VIDEODRIVER": "dummy"})` from the repo root with a 60-second timeout:
  - `test_play_headless_smoke`: exit code 0 for `--policy greedy` and `--policy random`.
  - **Review Focus 5:** `test_play_rl_without_model_falls_back`: `--policy rl --frames 5` exits 0 and its output contains the word `rl`.
  - `test_play_unknown_policy_is_rejected`: `--policy bogus` exits non-zero (argparse `choices`).

- [x] **Step 2: Run to verify failure.** `.venv/bin/pytest tests/test_play.py -q` fails (file missing).

- [x] **Step 3: Implement `play.py`.** Create the env with `render_mode="human"`; each loop: handle `pygame.event.get()` keys, if not paused `actions = policy.act(obs)` and `env.step`, set `env.hud_note`, call `env.render()`. When `env.agents` is empty, reset with the next seed. `--frames` counts loop iterations. Reset the policy with the episode seed at every reset.

- [x] **Step 4: Run to verify pass.** `.venv/bin/pytest tests/test_play.py -q` passes. Then run `.venv/bin/python play.py` yourself and check by eye: window opens, robots move to their targets, space pauses, P switches to random.

- [x] **Step 5: Commit.** `git add play.py tests/test_play.py && git commit -m "feat(warehouse): add live demo"`

---

# Stage 4: Ship the scripted version

### Task 6: README, GIF and checks

**Files:**
- Create: `scripts/record_demo.py` (kept local), `docs/multi_agent_warehouse/demo.gif`. The README is the root `README.md` (the plan originally said `multi_agent_warehouse/README.md`).

**Interfaces:**
- Produces: `scripts/record_demo.py` with `main() -> None` that runs a greedy episode (seed 0, 100 steps) with `render_mode="rgb_array"` and writes `docs/multi_agent_warehouse/demo.gif` with Pillow (`save_all=True`, `duration=100`, `loop=0`).

- [x] **Step 1: Write `scripts/record_demo.py`** and run it: `.venv/bin/python scripts/record_demo.py`. Open the GIF and check it plays and is under 3 MB.

- [x] **Step 2: Write `multi_agent_warehouse/README.md`:** a one-paragraph description, the GIF, install and run commands (`pip install -e .`, `python play.py`, `python -m multi_agent_warehouse.evaluate`, `pytest`), the evaluation table copied from Task 4's real output, and the design in five lines (ParallelEnv, two-rule move resolution, 8-value observation, offscreen-first renderer, policies are observation-only). State plainly that the larger project is `warehouse_rl/` and link `DESIGN.md`.

- [x] **Step 3: Add one line to the root `README.md`** linking to `multi_agent_warehouse/README.md`.

- [x] **Step 4: Run the CI-equivalent checks.** `.venv/bin/ruff check . && .venv/bin/ruff format --check . && SDL_VIDEODRIVER=dummy .venv/bin/pytest -q`. Everything passes, including the original `warehouse_rl` tests. Fix lint or format issues in the mini files only.

- [x] **Step 5: Commit.** `git add -A multi_agent_warehouse scripts docs/multi_agent_warehouse README.md && git commit -m "docs(warehouse): add README and demo GIF"`

**Stage 4 is a releasable stopping point: the scripted version is complete.**

---

# Stage 5: RL (stretch)

### Task 7: Dependency spike and training module

**Files:**
- Create: `multi_agent_warehouse/train.py` (a module, not a root script, so tests can import it under a plain `pytest` run)
- Modify: `pyproject.toml` (new extra `rl`)
- Test: `tests/test_rl.py`

**Interfaces:**
- Consumes: `MultiAgentWarehouseEnv(shaping=True)`.
- Produces: `multi_agent_warehouse/train.py` (run as `python -m multi_agent_warehouse.train`) with `train(total_steps: int = 300_000, n_envs: int = 8, seed: int = 0, out: str = "models/ppo_multi_agent_warehouse") -> Path` (returns the path of the saved zip) and `main()` with flags `--steps`, `--envs`, `--seed`, `--out`.

- [ ] **Step 1: Add the extra.** In `pyproject.toml` under `[project.optional-dependencies]` add `rl = ["stable-baselines3>=2.3", "supersuit>=3.9", "torch>=2.2"]`. Run `.venv/bin/pip install -e ".[rl]"`. If resolution or import fails, try pinning compatible versions once; if it still fails, stop and report (see the spec's gate).

- [ ] **Step 2: Write the failing test** in `tests/test_rl.py` with `pytest.importorskip("stable_baselines3")` at the top (so CI without the extra skips it): `test_train_saves_a_loadable_model(tmp_path)` imports `from multi_agent_warehouse.train import train` and calls `train(total_steps=512, n_envs=2, out=str(tmp_path / "m"))`, asserts the returned path exists and ends with `.zip`, and loads it with `PPO.load`.

- [ ] **Step 3: Run to verify failure.** `.venv/bin/pytest tests/test_rl.py -q` fails (import error).

- [ ] **Step 4: Implement `multi_agent_warehouse/train.py`.** Wrap the env with `supersuit.pettingzoo_env_to_vec_env_v1`, then `supersuit.concat_vec_envs_v1(vec, n_envs, num_cpus=1, base_class="stable_baselines3")`; build `PPO("MlpPolicy", vec_env, seed=seed, verbose=0)`; `learn(total_steps)`; save; make the output directory if missing. Import SB3 and SuperSuit inside `train()` so importing the module does not need them.

- [ ] **Step 5: Run the spike.** `.venv/bin/python -m multi_agent_warehouse.train --steps 2048 --out runs/spike/ppo` finishes and writes `runs/spike/ppo.zip`; then `.venv/bin/pytest tests/test_rl.py -q` passes. **Gate:** if either fails and the fix is not a small version change, stop here and ask.

- [ ] **Step 6: Commit.** `git add pyproject.toml multi_agent_warehouse/train.py tests/test_rl.py && git commit -m "feat(warehouse): add PPO training script"`

### Task 8: Train, wire in, report

**Files:**
- Create: `models/ppo_multi_agent_warehouse.zip`
- Modify: `multi_agent_warehouse/agents.py` (`RLPolicy`, `MODEL_PATH`, `available_policies()`), `multi_agent_warehouse/evaluate.py`, `play.py`, the root `README.md`
- Test: `tests/test_rl.py`

**Interfaces:**
- Produces in `agents.py`: `MODEL_PATH = Path("models/ppo_multi_agent_warehouse.zip")`; `class RLPolicy` with `__init__(self, model_path: str | Path = MODEL_PATH)` (imports SB3 lazily), `reset(seed)`, and `act(observations)` that stacks observations in sorted-id order, calls `model.predict(batch, deterministic=False)` once, and returns `dict[str, int]`; `available_policies() -> dict[str, Callable[[], Policy]]` returning `POLICIES` plus `"rl"` when `MODEL_PATH` exists and SB3 imports. `play.py` and `evaluate.main` use `available_policies()` instead of `POLICIES`.

- [ ] **Step 1: Write the failing tests** appended to `tests/test_rl.py`: `test_rl_policy_acts_for_all_robots(tmp_path)` trains a 512-step model into `tmp_path`, builds `RLPolicy(path)`, calls `act` on a real observation dict and asserts one valid action (0..4) per robot. `test_available_policies_without_model(monkeypatch)` points `MODEL_PATH` at a missing file and asserts `"rl"` is absent while `"random"` and `"greedy"` are present.

- [ ] **Step 2: Run to verify failure**, then implement `RLPolicy` and `available_policies()`, switch `play.py` and `evaluate.main` over, and run the tests until they pass.

- [ ] **Step 3: Train for real.** `.venv/bin/python -m multi_agent_warehouse.train` (300,000 steps). Evaluate with `.venv/bin/python -m multi_agent_warehouse.evaluate`. **Acceptance:** RL mean deliveries is at least twice random's. If not, retrain with more steps (`--steps 1000000`) and, if still short, with a different shaping coefficient (`SHAPING_SCALE` in `config.py`; exposing it as an env keyword is fine). Record which settings produced the committed model.

- [ ] **Step 4: Check the demo.** `.venv/bin/python play.py --policy rl` runs the learned policy by eye; `P` cycles random, greedy, rl. Delete the model temporarily and confirm `play.py --policy rl --frames 5` still exits 0 and prints a one-line hint (this is the Review Focus 5 test, now with the real code path).

- [ ] **Step 5: Update the root `README.md`** with the real three-row evaluation table, the training command, the settings used, and an honest sentence about where RL lands against greedy and what it was not tested on (one layout, 3 robots, no generalization).

- [ ] **Step 6: Final checks and commit.** `.venv/bin/ruff check . && .venv/bin/ruff format --check . && SDL_VIDEODRIVER=dummy .venv/bin/pytest -q` passes; check `models/ppo_multi_agent_warehouse.zip` is under 2 MB. `git add models multi_agent_warehouse play.py tests README.md && git commit -m "feat(warehouse): add trained PPO agent and wire it into demo and evaluation"`.
