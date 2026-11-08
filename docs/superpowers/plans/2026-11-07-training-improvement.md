# Training Improvement & Experiment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a randomly seeded, multi-seed experiment pipeline that trains the warehouse PPO agent under eight kinds of improvement and produces CSV data and graphs of how the model improves during training.

**Architecture:** `train.py` gains a `TrainConfig` and a `fit()` that logs and checkpoints. A new `experiment.py` runs variants x random seeds, evaluates every checkpoint on one shared random evaluation block, and writes CSVs. A new `plots.py` turns the CSVs into figures and `report.md`. The environment gains optional reward, observation and layout parameters whose defaults reproduce today's behaviour exactly.

**Tech Stack:** Python 3.11 (`.venv`), PettingZoo, Stable-Baselines3 2.9.2, SuperSuit 3.11.0, `sb3-contrib` (new), `matplotlib` (new), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-07-training-improvement-design.md`

## Global Constraints

- Python 3.10+ syntax; ruff rules `E, F, I, B, UP`, line length 100; `ruff format --check` clean.
- Every class and function follows `docs/code-style.md`: PythonTA docstrings with `Preconditions:` and, for classes, `Attributes:` and `Representation Invariants:`; `@check_contracts` on classes and public functions in modules that already use it.
- Tests run headless (`tests/conftest.py` sets `SDL_VIDEODRIVER=dummy`, `WAREHOUSE_CONTRACTS=1`). Run with `.venv/bin/python -m pytest`.
- SB3, SuperSuit, torch, sb3-contrib and matplotlib are optional extras: import them inside functions, and tests use `pytest.importorskip`.
- Evaluation always uses `shaping=False`. Defaults of `MultiAgentWarehouseEnv()` and `python -m multi_agent_warehouse.evaluate` (seeds 1000-1099) must not change behaviour or numbers.
- Training seeds are random integers in 0-999 unless `--seed(s)` is given, and are always written to `manifest.json`. Evaluation seed blocks are random per experiment, shared inside it, and disjoint from the training seeds.
- Commit messages follow `type(warehouse): summary`, one logical change each, **with no `Co-Authored-By` or Claude attribution trailer**.
- `runs/` and `docs/` are gitignored: results stay local unless force-added (`git add -f`) as this plan says.

## Review Focus

- **Default path must not drift:** the README table depends on identical RNG use and rewards when no new option is passed (Task 2 pins a greedy and a random episode; Task 4 pins `evaluate_with_ci`).
- **Same seed, same model:** two `fit` calls with one seed must give identical weights, or "random seeds" cannot be reproduced (Task 3 `test_same_seed_gives_identical_weights`).
- **Interrupted experiments:** a rerun must finish missing seeds without retraining or duplicating rows for finished ones (Task 5 `test_interrupted_run_is_completed_on_rerun`).
- **Degenerate plots:** one seed, one checkpoint, or a missing baseline must still produce a figure, not NaN bands or a crash (Task 6).
- **Shape mismatches:** the committed 8-value model on a `rich` env must fail with a readable message, and recurrent state must not leak between episodes or robot counts (Tasks 4, 9, 10).

---

### Task 1: Branch, dependencies, seed utilities

**Files:**
- Modify: `pyproject.toml` (add `sb3-contrib>=2.3` to the `rl` extra; new extra `plot = ["matplotlib>=3.8"]`)
- Create: `multi_agent_warehouse/seeds.py`
- Test: `tests/test_seeds.py`

**Interfaces:**
- Produces: `draw_seeds(n: int) -> list[int]` (n distinct ints in `[0, 1000)`, so `n <= 1000`; training seeds are always 0-999); `draw_eval_base(n_episodes: int = 100) -> int` (a start such that `base .. base + n_episodes - 1` fits in `[10_000, 2**31)`).

- [ ] **Step 1:** Ask the user to commit or stash the README edits on `chores/readme-ppo-update`; then `git switch main && git pull && git switch -c feat/training-experiments`.
- [ ] **Step 2:** Install the extras: `.venv/bin/pip install -e ".[dev,rl,plot]"`. Expected: `.venv/bin/python -c "import sb3_contrib, matplotlib"` prints nothing.
- [ ] **Step 3: Write tests** `test_draw_seeds_returns_n_distinct_ints_in_range` (all in 0-999), `test_draw_seeds_can_return_every_seed` (`draw_seeds(1000)` sorted equals `list(range(1000))`), `test_draw_seeds_rejects_more_than_1000`, `test_draw_seeds_differs_between_calls` (two draws of 8 are not equal), `test_draw_eval_base_leaves_room_for_the_block` (`10_000 <= base` and `base + 100 <= 2**31`), `test_eval_block_never_overlaps_training_seeds`.
- [ ] **Step 4:** Run `pytest tests/test_seeds.py -v`; expected FAIL (module missing).
- [ ] **Step 5: Implement** both functions in `seeds.py` with `secrets.randbelow` (OS entropy, not NumPy's global state).
- [ ] **Step 6:** Run the tests; expected PASS. Commit `feat(warehouse): add random seed utilities and plotting deps`.

### Task 2: Environment reward parameters and truncation flag

**Files:**
- Modify: `multi_agent_warehouse/env.py` (`__init__`, `_reward`, `step`)
- Test: `tests/test_env.py`, `tests/test_rl.py`

**Interfaces:**
- Produces: `MultiAgentWarehouseEnv(..., *, collision_penalty: float = COLLISION_PENALTY, shaping_scale: float = SHAPING_SCALE)`; every `infos[agent]` carries `"TimeLimit.truncated": bool`, True only on the step that truncates.

- [ ] **Step 1: Pin the current behaviour.** Before editing, run `run_episode(MultiAgentWarehouseEnv(), GreedyPolicy(), 1000)` and the same with `RandomPolicy()` (from `evaluate.py`) and write the two result dicts into the test file as constants.
- [ ] **Step 2: Write tests** `test_default_episodes_match_pinned_results` (the two constants), `test_custom_collision_penalty_is_used` (penalty -0.5: a colliding robot's reward is `STEP_PENALTY - 0.5`), `test_custom_shaping_scale_is_used` (shaping on, scale 0.1, one step toward the target pays `STEP_PENALTY + 0.1`), `test_truncation_flag_only_on_last_step` (False on step 199, True on step 200), and in `test_rl.py` `test_vec_env_exposes_truncation_flag_and_terminal_observation` (build the vec env as `train` does, step 200 times, last infos contain both keys).
- [ ] **Step 3:** Run them; the new ones fail, the pinned test passes.
- [ ] **Step 4: Implement:** store the two values as attributes, use them in `_reward`, add the info key in `step`. Raise `ValueError` for `collision_penalty > 0` or `shaping_scale < 0`.
- [ ] **Step 5:** Run `pytest tests/test_env.py tests/test_rl.py -v`; all PASS. Commit `feat(warehouse): configurable rewards and truncation flag`.

### Task 3: TrainConfig, logging, checkpoints, seeded determinism

**Files:**
- Modify: `multi_agent_warehouse/train.py`, `tests/test_rl.py`

**Interfaces:**
- Consumes: Task 1 `draw_seeds`; Task 2 env arguments.
- Produces: frozen dataclass `TrainConfig(total_steps=300_000, n_envs=8, seed=0, n_steps=128, batch_size=256, learning_rate=3e-4, gamma=0.99, ent_coef=0.0, net_arch=(64, 64), algo="ppo", bootstrap_truncation=True, robot_counts=(3,), env_kwargs=<empty dict>, checkpoint_steps=())`; `fit(cfg: TrainConfig, out: Path, log_dir: Path | None = None, verbose: int = 0) -> Path` (final zip path); `train(...)` keeps its current signature and calls `fit`; `CheckpointAtSteps(BaseCallback)`.
- Produces on disk under `log_dir`: `progress.csv`, `config.json`, `checkpoints/step_<timesteps:09d>.zip` (saved at the first rollout end at or after each requested step, named by the real timestep count).

- [ ] **Step 1: Write tests** `test_train_config_roundtrips_through_json`, `test_fit_writes_checkpoints_progress_and_config` (cfg with `n_envs=2`, `total_steps=3072`, `checkpoint_steps=(1024, 2048)`: two checkpoint files; `progress.csv` has `time/total_timesteps`, `rollout/ep_rew_mean`, `train/entropy_loss`, `train/value_loss`, `train/explained_variance`), `test_same_seed_gives_identical_weights` (two fits, seed 7, 1024 steps, `torch.equal` on every state-dict tensor), `test_different_seeds_give_different_weights`, `test_hyperparameters_reach_the_model` (`ent_coef=0.01`, `gamma=0.95`, `net_arch=(32, 32)` visible on the loaded model and its first policy layer width), `test_disabling_bootstrap_removes_the_flag` (infos lack `TimeLimit.truncated`), `test_main_draws_a_random_seed_when_none_is_given` (monkeypatch `fit`; two `main([])` calls pass different seeds and print them).
- [ ] **Step 2:** Run; expect FAIL. The existing `test_train_saves_a_loadable_model` must keep passing unchanged.
- [ ] **Step 3: Implement** `fit`: build the env list from `robot_counts` cycled over `n_envs`, wrap with `supersuit.pettingzoo_env_to_vec_env_v1` and SuperSuit's `ConcatVecEnv`, then `SB3VecEnvWrapper`; seed with `set_random_seed` and `venv.reset(seed=...)` **before** adding `VecMonitor` (which hides `.venv`); a private `VecEnvWrapper` drops the truncation key when `bootstrap_truncation=False`; pass `policy_kwargs={"net_arch": ...}`; attach `stable_baselines3.common.logger.configure(log_dir, ["csv"])`; write `config.json` with `dataclasses.asdict`. If `ConcatVecEnv` cannot take mixed vec envs in SuperSuit 3.11.0, fall back to training each distinct robot count in sequential phases with `model.set_env` and `learn(reset_num_timesteps=False)`.
- [ ] **Step 4:** CLI: add `--seed` (default `None`, drawn with `draw_seeds(1)[0]` and printed), `--ent-coef`, `--gamma`, `--lr`, `--net-arch`, `--log-dir`.
- [ ] **Step 5:** Run `pytest tests/test_rl.py -v`; all PASS. Commit `feat(warehouse): TrainConfig with logging, checkpoints and random seeds`.

### Task 4: Policy and evaluation plumbing

**Files:**
- Modify: `multi_agent_warehouse/agents.py`, `multi_agent_warehouse/evaluate.py`
- Test: `tests/test_agents.py`, `tests/test_rl.py`, `tests/test_evaluate.py` (new)

**Interfaces:**
- Produces: `RLPolicy(model_path=MODEL_PATH, deterministic: bool = False, algo: str = "ppo")`; `evaluate_episodes(make_policy, seeds: Sequence[int], n_robots=DEFAULT_N_ROBOTS, env_kwargs: Mapping | None = None) -> list[dict[str, int]]`; `summarise(results: Sequence[dict[str, int]]) -> dict[str, tuple[float, float]]` (mean and 95% half-width, 0.0 for one episode). `evaluate_with_ci` is rebuilt on these and returns the same numbers.

- [ ] **Step 1: Pin** `evaluate_with_ci(GreedyPolicy, n_episodes=5)` as constants before editing.
- [ ] **Step 2: Write tests** `test_evaluate_with_ci_unchanged`, `test_evaluate_episodes_returns_one_row_per_seed`, `test_summarise_single_episode_has_zero_half_width`, `test_deterministic_rl_policy_repeats_actions`, `test_rl_policy_rejects_wrong_observation_size` (a 14-value observation raises `ValueError` whose message contains `expects 8`; skipped if the committed model is missing).
- [ ] **Step 3:** Run; expect the new tests FAIL.
- [ ] **Step 4: Implement.** `RLPolicy.act` checks `batch.shape[1] == model.observation_space.shape[0]` and passes `deterministic` to `predict`. Relax the `Policy.act` precondition from `len(obs) == 8` to `len(obs) >= 8` (the first 8 values keep their meaning in every observation mode).
- [ ] **Step 5:** Run `pytest -v`; all PASS. Commit `feat(warehouse): deterministic RL policy and episode-level evaluation`.

### Task 5: Experiment runner

**Files:**
- Create: `multi_agent_warehouse/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `TrainConfig`, `fit`, `RLPolicy`, `evaluate_episodes`, `summarise`, `draw_seeds`, `draw_eval_base`.
- Produces: `VARIANTS: dict[str, TrainConfig]` containing `legacy` (committed settings, `bootstrap_truncation=False`, 300k steps, checkpoints every 50k) and `base` (`bootstrap_truncation=True`, 3M steps, checkpoints at 50k, 100k, 200k, 300k, 500k, 1M, 2M, 3M); `run_experiment(name: str, variants: Sequence[str], n_seeds: int = 5, total_steps: int | None = None, train_seeds: Sequence[int] | None = None, eval_seed_base: int | None = None, n_eval_episodes: int = 100, out_root: Path = Path("runs/experiments")) -> Path`; `evaluate_run(run_dir: Path, cfg: TrainConfig, eval_seeds: Sequence[int], modes: Sequence[str] = ("sampled", "deterministic")) -> list[dict]`.
- CSV schemas: `eval_curve.csv` = `variant,seed,timesteps,mode,mean_deliveries,ci_deliveries,mean_collisions,ci_collisions,n_episodes`; `baselines.csv` = `policy,mean_deliveries,ci_deliveries,mean_collisions,ci_collisions,n_episodes`; `manifest.json` keys `name, created, git_commit, versions, train_seeds, eval_seed_base, n_eval_episodes, variants`.
- CLI: `python -m multi_agent_warehouse.experiment run --name N --variants a,b --seeds K [--steps S] [--train-seeds 1,2] [--eval-seed-base B]`; `... plot --name N` is added in Task 6.

- [ ] **Step 1: Write tests** (tiny budgets: `total_steps=3072`, 2 seeds, `n_eval_episodes=2`, `n_envs=2`) `test_run_experiment_writes_manifest_curve_and_baselines`, `test_train_seeds_are_random_unless_given` (two runs differ; explicit seeds are used), `test_train_seeds_avoid_the_eval_block`, `test_manifest_records_commit_and_versions`, `test_eval_curve_has_a_row_per_checkpoint_seed_and_mode`, `test_baselines_use_the_same_eval_seeds` (greedy row equals `summarise(evaluate_episodes(GreedyPolicy, eval_seeds))`), `test_interrupted_run_is_completed_on_rerun` (delete one seed directory after a run, rerun with the same name: only that seed is retrained via a counting monkeypatch of `fit`, and the CSV has no duplicate `(variant, seed, timesteps, mode)` keys).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement.** Training seeds and eval base are drawn once, written to the manifest first, and read back from it on a rerun. A run is "complete" when `final.zip` and its rows in `eval_curve.csv` exist; evaluation appends per-checkpoint rows and skips keys already present. Evaluate `RLPolicy` objects once per checkpoint and reuse them for all episodes (one model load, not one per episode).
- [ ] **Step 4:** Run the tests; PASS. Commit `feat(warehouse): multi-seed experiment runner`.

### Task 6: Plots and report

**Files:**
- Create: `multi_agent_warehouse/plots.py`; modify `experiment.py` (`plot` subcommand)
- Test: `tests/test_plots.py`

**Interfaces:**
- Produces (all import matplotlib inside the function and use the Agg backend): `seed_band(values: Sequence[float]) -> tuple[float, float, float]` (mean, lower, upper of a 95% t-interval; `(v, v, v)` for one value); `plot_learning_curve(exp_dir: Path, metric: str, out: Path, mode: str = "sampled") -> Path` (x = timesteps on a log axis; thin line per seed, bold mean, shaded band, dashed random/greedy lines); `plot_sampled_vs_deterministic(exp_dir, variant, out)`; `plot_training_diagnostics(exp_dir, variant, out)` (2x2: ep reward, entropy, value loss, explained variance vs timesteps, mean over seeds); `plot_variant_comparison(exp_dir, out)` (final-checkpoint deliveries per variant, error bars = seed std, sorted); `plot_tradeoff(exp_dir, out)` (deliveries vs collisions, one dot per seed, random/greedy markers); `make_report(exp_dir: Path) -> Path` (`report.md` with a final-checkpoint table per variant and links to each figure); `promote(exp_dir, names, dest=Path("docs/multi_agent_warehouse")) -> list[Path]`.

- [ ] **Step 1: Write tests** against synthetic CSVs in `tmp_path`: `test_seed_band_values` (`[1, 2, 3]` gives mean 2 and half-width about 2.484), `test_seed_band_single_value_has_no_nan`, `test_learning_curve_writes_png`, `test_learning_curve_with_one_seed_and_one_checkpoint`, `test_learning_curve_without_baselines_file`, `test_variant_comparison_orders_by_score`, `test_make_report_links_every_figure_it_made`, `test_promote_copies_named_figures`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement** with `csv` plus `numpy` (no pandas). The `plot` subcommand writes every applicable figure into `<exp>/plots/` and then the report; figures whose input data is absent (for example diagnostics without `progress.csv`) are skipped, not errors.
- [ ] **Step 4:** Run the tests; PASS. Smoke check: `experiment run --name smoke --variants base --seeds 2 --steps 3072` then `experiment plot --name smoke`, and open `runs/experiments/smoke/plots/*.png` to confirm axes, legend and baselines look right. Commit `feat(warehouse): experiment plots and report`.

### Task 7: Benchmark, then E0 and E1 (steps 1, 2, 3)

**Files:** none new; results in `runs/experiments/e0-legacy/`, `runs/experiments/e1-base/`; findings in `docs/multi_agent_warehouse/results/FINDINGS.md`.

- [ ] **Step 1: Benchmark** `time .venv/bin/python -m multi_agent_warehouse.train --steps 60000 --out runs/bench/m`; record steps/second. Estimated E1 hours = `5 * 3_000_000 / steps_per_second / 3600`; if over 3, use `--steps 1000000` for E1 and note it in `FINDINGS.md`.
- [ ] **Step 2: E0** (about the committed setup, 5 random seeds): `.venv/bin/python -m multi_agent_warehouse.experiment run --name e0-legacy --variants legacy --seeds 5`. Expected: five `seed_*` directories, `eval_curve.csv` with rows at 50k..300k.
- [ ] **Step 3: E1:** `... run --name e1-base --variants base --seeds 5 --steps 3000000`, then `... plot --name e0-legacy` and `... plot --name e1-base`.
- [ ] **Step 4: Read the figures** (open each PNG) and write to `FINDINGS.md`: (a) the spread of `legacy` at 300k against the committed 51.07; (b) whether truncation bootstrapping moves the 300k score (compare E0 and E1 at 300k; different eval blocks, so compare with caution and say so); (c) the step count where deliveries stop improving; (d) the sampled-vs-deterministic gap; (e) whether collisions fall as deliveries rise.
- [ ] **Step 5:** `git add -f docs/multi_agent_warehouse/results/FINDINGS.md` plus the E1 learning-curve and diagnostics PNGs; commit `docs: record seed spread and learning curves`.

### Task 8: Hyperparameter and reward sweep (steps 4, 5)

**Files:**
- Modify: `multi_agent_warehouse/experiment.py` (registry), `tests/test_experiment.py`

**Interfaces:**
- Produces variants in `VARIANTS`, each equal to `base` except the named field, 1M steps: `ent_0.01` (`ent_coef=0.01`), `shape_0` / `shape_0.05` / `shape_0.1` (`env_kwargs={"shaping_scale": ...}`; `shape_0` also sets shaping off via scale 0), `gamma_0.95`, `gamma_0.995`, `net_128` (`net_arch=(128, 128)`), `coll_0.3` / `coll_1.0` (`env_kwargs={"collision_penalty": -0.3}` / `-1.0`).

- [ ] **Step 1: Write test** `test_sweep_variants_differ_from_base_in_exactly_one_field` (compare `asdict`, flatten `env_kwargs`, expect one differing key per variant).
- [ ] **Step 2:** Run; FAIL. Implement the registry entries; run; PASS. Commit `feat(warehouse): sweep variants`.
- [ ] **Step 3: Run E2** (include `base` as the control so every comparison is paired on one eval block): `... run --name e2-sweep --variants base,ent_0.01,shape_0,shape_0.05,shape_0.1,gamma_0.95,gamma_0.995,net_128,coll_0.3,coll_1.0 --seeds 3 --steps 1000000`; then `plot`.
- [ ] **Step 4:** Add to `FINDINGS.md` a ranked table (final deliveries and collisions per variant, mean +/- seed std) and mark each knob as "helps", "no effect" (difference under one seed std) or "hurts". Commit with `git add -f`.

### Task 9: Richer observation (step 6)

**Files:**
- Modify: `config.py` (`OBSERVATION_MODES = ("basic", "rich")`, `OBS_SIZES = {"basic": 8, "rich": 14}`), `env.py`, `experiment.py` (variant `rich_obs`)
- Test: `tests/test_env.py`, `tests/test_agents.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces: `MultiAgentWarehouseEnv(..., *, observation: str = "basic")`. `rich` = the 8 basic values, then 4 flags (up, down, left, right neighbour holds a robot), then the offset (d_row, d_col) to the nearest other robot divided by the same scale, ties broken by smaller Manhattan distance then lower robot id, `(0, 0)` when there is only one robot.

- [ ] **Step 1: Write tests** `test_rich_observation_has_14_values_and_basic_prefix`, `test_rich_robot_flags_separate_robots_from_shelves` (robot beside a shelf has its basic flag 1 and robot flag 0; robot beside a robot has both 1), `test_nearest_robot_offset_and_tie_break`, `test_single_robot_rich_offset_is_zero`, `test_invalid_observation_mode_raises`, `test_greedy_policy_acts_identically_on_rich_observations`, `test_rich_obs_variant_sets_observation_kwarg`.
- [ ] **Step 2:** Run; FAIL. Implement (observation spaces become `Box(-1, 1, (OBS_SIZES[mode],))`); run; PASS.
- [ ] **Step 3:** Confirm `pytest tests/test_agents.py -k rl_policy_rejects_wrong_observation_size` still passes. Commit `feat(warehouse): optional rich observation`.

### Task 10: Recurrent policy (step 7)

**Files:**
- Modify: `train.py` (`algo="recurrent_ppo"` builds `RecurrentPPO("MlpLstmPolicy", ...)`), `agents.py` (`RLPolicy` keeps per-robot LSTM state when `algo="recurrent_ppo"`), `experiment.py` (variants `lstm`, `rich_lstm`)
- Test: `tests/test_rl.py`

**Interfaces:**
- `RLPolicy.reset(seed)` clears the recurrent state; `act` passes `state` and `episode_start` (True on the first call after `reset`) to `predict` and stores the returned state. Robots are batched in sorted id order, which matches training order.

- [ ] **Step 1: Write tests** (skip without `sb3_contrib`) `test_recurrent_fit_saves_a_loadable_model`, `test_recurrent_policy_state_changes_between_steps`, `test_recurrent_state_clears_on_reset` (after `reset(0)`, a deterministic policy's first action equals that of a freshly built policy), `test_recurrent_policy_handles_robot_count_change_after_reset` (3 robots then 4).
- [ ] **Step 2:** Run; FAIL. Implement; `RecurrentPPO` needs `batch_size` compatible with its sequence splitting, so verify against sb3-contrib and adjust the variant's `batch_size` if the first run raises. PASS.
- [ ] **Step 3: Run E3:** `... run --name e3-structure --variants base,rich_obs,lstm,rich_lstm --seeds 3 --steps 1000000`; `plot`. Add the ranked table to `FINDINGS.md`. Commit `feat(warehouse): recurrent PPO option`.

### Task 11: Randomised layouts and robot counts (step 8)

**Files:**
- Modify: `config.py` (`LAYOUTS: dict[str, tuple[str, ...]]` with `default`, `open`, `aisles`, `scatter`, `wide_gap`, `maze_lite`; `LAYOUT = LAYOUTS["default"]`; `TRAIN_LAYOUTS = ("default", "open", "aisles", "scatter")`; `HELDOUT_LAYOUTS = ("wide_gap", "maze_lite")`), `env.py`, `train.py`, `experiment.py`, `plots.py`
- Test: `tests/test_config.py`, `tests/test_env.py`, `tests/test_rl.py`, `tests/test_experiment.py`, `tests/test_plots.py`

**Interfaces:**
- Produces: `MultiAgentWarehouseEnv(..., *, layouts: tuple[str, ...] = ("default",))` (one is chosen per `reset`; with a single layout the RNG stream is unchanged); `evaluate_generalisation(exp_dir: Path, variant: str, layouts: Sequence[str], robot_counts: Sequence[int], n_episodes: int) -> list[dict]` (writes `generalisation.csv`: `variant,seed,layout,n_robots,mean_deliveries,ci_deliveries,mean_collisions,ci_collisions`); `plot_generalisation(exp_dir, out)` (heatmap, rows = layouts, columns = robot counts, one panel per variant); variants `rand` (`env_kwargs={"layouts": TRAIN_LAYOUTS}`, `robot_counts=(2, 3, 4, 5)`) and `fixed` (= `base`, named for the comparison).

- [ ] **Step 1: Write tests** `test_every_layout_is_rectangular_8x8_and_connected`, `test_every_layout_has_room_for_max_robots` (at least `2 * MAX_ROBOTS` free cells), `test_heldout_layouts_are_not_training_layouts`, `test_multi_layout_env_uses_every_layout_across_seeds`, `test_renderer_follows_the_active_layout` (shelves passed to the renderer match the layout after a reset that switches layouts), `test_mixed_robot_counts_vec_env_has_summed_agents` (`robot_counts=(2, 3, 4)`, `n_envs=3` gives 9 sub-environments), `test_generalisation_rows_cover_layout_and_count_grid`, `test_plot_generalisation_writes_png`. Re-run Task 2's pinned-episode test: it must still pass.
- [ ] **Step 2:** Run; FAIL. Implement; PASS. Commit `feat(warehouse): randomised layouts and robot counts`.
- [ ] **Step 3: Run E4:** `... run --name e4-generalise --variants fixed,rand --seeds 3 --steps 1000000`, then generalisation over all 6 layouts and robot counts 2-6 (100 episodes each) and `plot`. Record in `FINDINGS.md` the held-out gap for `fixed` against `rand`. Commit.

### Task 12: Tuned model, honest selection, README

**Files:**
- Modify: `experiment.py` (variant `tuned`, `select_and_test`), `evaluate.py` and `play.py` (read the model's `<name>.json` sidecar to build the env), `README.md`, `models/`
- Test: `tests/test_experiment.py`, `tests/test_evaluate.py`, `tests/test_play.py`

**Interfaces:**
- Produces: `select_and_test(exp_dir: Path, variant: str) -> dict` (picks the seed with the best final deliveries on a fresh random validation block, reports that seed's score on a second fresh random test block disjoint from both the validation block and all training seeds, writes `final.csv`); `policy_env_kwargs(model_path: Path) -> dict` (env kwargs from the sidecar, `{}` when it is absent).

- [ ] **Step 1: Write tests** `test_select_and_test_blocks_are_disjoint`, `test_select_and_test_reports_the_selected_seed_on_the_test_block`, `test_policy_env_kwargs_reads_sidecar`, `test_policy_env_kwargs_defaults_without_sidecar`, `test_evaluate_main_builds_rich_env_for_a_rich_model` (tiny trained model in `tmp_path`).
- [ ] **Step 2:** Run; FAIL. Implement; PASS. Commit `feat(warehouse): honest model selection and sidecar-aware evaluation`.
- [ ] **Step 3: Define `tuned`** from the "helps" rows in `FINDINGS.md` only (combine winners from E2, E3, E4; if the layout-randomised variant wins on held-out layouts but loses on the default map, record that trade-off and choose with the user). Run `... run --name e5-tuned --variants tuned,base --seeds 5 --steps 3000000` (or the Task 7 budget), `plot`, then `select_and_test`.
- [ ] **Step 4: Replace the committed model only if the test block shows `tuned` above `base` by more than the seed spread.** Move the old model to `models/ppo_v1_300k.zip`, copy the selected `final.zip` to `models/ppo_multi_agent_warehouse.zip` with its sidecar, then run `python -m multi_agent_warehouse.evaluate` (seeds 1000-1099) and update the README table.
- [ ] **Step 5: README:** promote the final figures (`promote`) into `docs/multi_agent_warehouse/`, embed the learning curve, sampled-vs-deterministic and generalisation plots, document the seed policy and the experiment commands, correct the line saying the committed model used seed 0 defaults only if the model changed, and tick the two roadmap items that are now answered. Run `pytest` and `ruff check . && ruff format --check .`; expected: all pass. Commit `docs: report improved training results`, then ask the user before pushing or opening a PR.

---

## Self-Review Notes

- **Coverage:** step 1 seeds -> Tasks 1, 3, 5, 7; step 2 deterministic -> Tasks 4, 7; step 3 longer training and checkpoints -> Tasks 3, 5, 7; step 4 hyperparameters and step 5 collision penalty -> Task 8; step 6 observation -> Task 9; step 7 recurrence -> Task 10; step 8 randomisation -> Task 11; graphs and data -> Tasks 5, 6, 7 plus every run task; randomised seeds -> Tasks 1, 3, 5, 12.
- **Consistency:** `TrainConfig.env_kwargs` carries `shaping_scale`, `collision_penalty`, `observation`, `layouts`, the same keys the env accepts; `RLPolicy(model_path, deterministic, algo)` is used identically in Tasks 4, 5, 10.
- **Risks to watch while executing:** `ConcatVecEnv` with mixed robot counts (fallback in Task 3), `RecurrentPPO` batch-size rules (Task 10), and throughput deciding the real step budgets (Task 7).
