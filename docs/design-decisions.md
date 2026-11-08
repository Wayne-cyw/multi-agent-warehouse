# Design Decisions: Multi-Agent Warehouse

Every design decision in this project, with the reason behind it, the alternatives that were
rejected, and the trade-off accepted. Read the [study notes](study-notes/README.md) first if a term
is unfamiliar (one topic per file; the glossary is [15](study-notes/15-glossary.md)).

**Sources:** `docs/superpowers/specs/2026-10-28-multi-agent-warehouse-design.md` (the spec),
`docs/superpowers/plans/2026-10-28-multi-agent-warehouse.md` (the plan), `pyproject.toml`, and the
installed PettingZoo 1.27.0 source. Where a reason is not stated in those documents and I am
inferring it, it is marked *(inferred)*.

> **Project status (2026-11-07): Stages 1 to 4 implemented and merged; Stage 5 Task 7 (the SuperSuit spike and
> `train.py`) done on branch `feat/ppo-training-script`; Task 8 (`RLPolicy`, the trained model) done on branch
> `feat/ppo-trained-agent` (stacked on Task 7).**
> The environment, rules, renderer, random and greedy policies, evaluation and demo exist, with 60
> passing tests and clean `ruff` (see the [status snapshot](study-notes/README.md#status-snapshot-as-of-2026-11-03)).
> D-41, D-42 and D-52 are now measured (see their entries); D-39, D-40, D-43, D-44 and D-53 still describe
> intended code.
> Update the "Status" lines as they are built or changed.

## How to read an entry

| Field | Meaning |
| --- | --- |
| **Status** | `Decided`: stated as a goal, constraint or rule in the spec, or accepted by you later in conversation (marked "accepted" with the date). `Unconfirmed`: listed in the spec under "Decisions made by me" and still waiting for your OK (none of the ones that matter for RL are left after 2026-11-03). `Implemented`: an implementation-level choice made in the plan and now built. `Plan`: an implementation-level choice not built yet. `Proposed`: raised in the study notes, **not** in the spec or plan; it needs your decision before it counts. |
| **Why** | The reasoning. |
| **Alternatives** | What else was possible. |
| **Trade-off** | What this choice costs. |

## Index

| # | Decision | Status |
| --- | --- | --- |
| **A. Scope and strategy** | | |
| [D-01](#d-01-a-new-small-package-instead-of-extending-warehouse_rl) | New small package instead of extending `warehouse_rl` | Decided (built and merged) |
| [D-02](#d-02-optimise-for-readability-not-completeness) | Optimise for readability, about 450 lines | Decided |
| [D-03](#d-03-explicit-non-goals) | Explicit non-goals | Decided |
| [D-04](#d-04-five-stages-each-ending-in-something-runnable) | Five stages, each ends runnable; RL is a stretch | Decided |
| [D-05](#d-05-anti-goals-as-guard-rails) | Anti-goals as guard rails | Decided |
| **B. Environment: world and interface** | | |
| [D-06](#d-06-pettingzoo-parallelenv-not-aec) | PettingZoo `ParallelEnv`, not AEC | Decided |
| [D-07](#d-07-one-fixed-8-x-8-layout-in-a-settings-file) | One fixed 8 x 8 layout in a settings file | Decided |
| [D-08](#d-08-robot-count-and-naming) | 1 to 6 robots, default 3, named `robot_i` | Decided |
| [D-09](#d-09-starts-and-targets) | Random distinct starts; one target per robot, resampled on delivery | Decided |
| [D-10](#d-10-five-discrete-actions) | `Discrete(5)` actions | Decided |
| [D-11](#d-11-an-8-number-local-observation) | 8-number local observation instead of grid windows | Decided (confirmed 2026-11-03) |
| [D-12](#d-12-normalise-by-maxh-w---1) | Normalise by `max(H, W) - 1` | Decided |
| [D-13](#d-13-neighbour-flags-include-other-robots) | Neighbour-blocked flags include other robots | Decided |
| **C. Physics and rewards** | | |
| [D-14](#d-14-move-resolution-as-two-rules-plus-a-fixed-point) | Move resolution: two rules plus a fixed point, as a pure function | Decided (confirmed 2026-11-03) |
| [D-15](#d-15-four-robot-rotations-are-allowed) | Four-robot rotations are allowed | Decided (confirmed 2026-11-03) |
| [D-16](#d-16-robots-held-up-behind-a-conflict-are-penalised-too) | Robots held up behind a conflict are penalised too | Decided (confirmed 2026-11-03; revisit if RL learns oddly) |
| [D-17](#d-17-bumping-a-wall-or-shelf-is-not-a-collision) | Wall and shelf bumps are not collisions | Decided |
| [D-18](#d-18-step-order-and-observation-last) | Fixed step order; observations built last | Decided |
| [D-19](#d-19-individual-rewards) | Individual rewards | Decided |
| [D-20](#d-20-reward-magnitudes) | Reward magnitudes: +1.0, -0.01, -0.1 | Decided |
| [D-21](#d-21-potential-based-shaping-off-by-default-never-used-in-evaluation) | Shaping: potential-based, off by default, never in evaluation | Decided |
| **D. Episode, state and robustness** | | |
| [D-22](#d-22-no-terminations-truncate-everyone-together) | No terminations; truncate everyone together at 200 | Decided |
| [D-23](#d-23-step-after-the-episode-ended-returns-empty-dicts) | `step()` after the end returns five empty dicts | Implemented |
| [D-24](#d-24-validate-inputs-before-changing-state) | Validate inputs with `ValueError` before changing state | Implemented |
| [D-25](#d-25-one-seeded-generator-no-global-randomness) | One seeded generator, no global RNG | Decided |
| [D-26](#d-26-public-state-attributes-and-info-keys) | Public state attributes and `info` keys | Decided |
| [D-27](#d-27-cache-the-spaces-with-lru_cache) | Cache spaces with `lru_cache` | Implemented |
| [D-28](#d-28-iterate-robots-in-sorted-id-order) | Iterate robots in sorted-id order | Implemented |
| **E. Rendering** | | |
| [D-29](#d-29-draw-offscreen-first) | Draw offscreen first, window is optional | Decided |
| [D-30](#d-30-import-pygame-lazily-and-only-in-the-renderer) | Import Pygame lazily, only in the renderer | Decided |
| [D-31](#d-31-fixed-visual-constants) | Fixed visual constants | Decided |
| [D-32](#d-32-render_modenone-means-do-nothing) | `render_mode=None` means do nothing | Decided |
| **F. Policies, evaluation and demo** | | |
| [D-33](#d-33-a-two-method-policy-interface-observations-only) | Two-method policy interface, observations only | Decided |
| [D-34](#d-34-the-greedy-rule-with-a-random-sidestep) | Greedy rule with a random sidestep | Decided |
| [D-35](#d-35-a-tiny-fair-evaluation) | Paired evaluation: 100 episodes, 95% confidence intervals | Decided (revised 2026-11-03) |
| [D-36](#d-36-entry-points-and-demo-controls) | Entry points and demo controls | Decided (built and merged) |
| [D-37](#d-37-the-demo-degrades-gracefully) | The demo degrades gracefully | Decided |
| [D-38](#d-38-no-keyboard-controlled-robot-yet) | No keyboard-controlled robot yet | Unconfirmed |
| **G. Reinforcement learning (stretch)** | | |
| [D-39](#d-39-use-stable-baselines3-ppo-do-not-reimplement-it) | Use Stable-Baselines3 PPO | Decided |
| [D-40](#d-40-parameter-sharing-through-supersuit) | Parameter sharing through SuperSuit | Decided |
| [D-41](#d-41-rl-dependencies-are-an-optional-extra-imported-lazily) | RL dependencies optional and lazily imported | Decided |
| [D-42](#d-42-a-spike-with-a-stop-and-ask-gate) | Spike with a stop-and-ask gate | Decided |
| [D-43](#d-43-sample-actions-at-inference-time) | Sample actions at inference time | Decided |
| [D-44](#d-44-commit-the-model-and-report-honestly) | Commit the model; fixed acceptance bar; report honestly | Decided |
| [D-52](#d-52-set-ppos-rollout-size-and-training-budget-explicitly) | Set PPO's rollout size and training budget explicitly | Decided (accepted) |
| [D-53](#d-53-generalization-is-a-separate-stage-6-built-one-level-at-a-time) | Generalization as a separate Stage 6, one level at a time | Decided (accepted) |
| **H. Tooling and process** | | |
| [D-45](#d-45-python-310-with-the-project-venv) | Python 3.10+, project venv | Decided |
| [D-46](#d-46-ruff-lint-and-format-exactly-as-ci-does) | Ruff exactly as CI runs it | Decided |
| [D-47](#d-47-test-first-flat-test-layout) | Test first; flat test layout | Implemented |
| [D-48](#d-48-headless-testing-with-sdl_videodriverdummy) | Headless tests with `SDL_VIDEODRIVER=dummy` | Implemented |
| [D-49](#d-49-type-hints-and-docstrings-on-the-public-surface) | Type hints and docstrings on the public surface | Implemented |
| [D-50](#d-50-pillow-for-the-readme-gif) | Pillow for the README GIF | Decided (confirmed 2026-11-03) |
| [D-51](#d-51-branch-and-commit-conventions) | Branch and commit conventions | Implemented |

Section I at the end lists **open questions, risks and inconsistencies** found while writing this.

---

## A. Scope and strategy

### D-01: A new small package instead of extending `warehouse_rl`
- **Status:** Decided (built and merged). New package `multi_agent_warehouse/`; this repository
  contains only that package. The `warehouse_rl` project this was split from is not in this repo.
- **Why:** `warehouse_rl` is rigorous but large: three layouts, five baselines, a deadlock detector,
  a statistics harness and 17 test files. A reviewer cannot take it in at a glance. A clone-and-run
  project that shows a working multi-agent environment in 30 seconds has a different audience.
  (The README no longer points at that project.)
- **Alternatives:** trim `warehouse_rl` in place; add a "lite" mode to it.
- **Trade-off:** some duplicated ideas across two packages, in exchange for zero risk to the
  existing project and a codebase small enough to read.
- **Note:** `warehouse_rl/` never lived in this repository, so nothing here depends on it.

### D-02: Optimise for readability, not completeness
- **Status:** Decided. Target is about 450 lines of library code, readable in one sitting.
- **Why:** The project's purpose is to *show* a correct `ParallelEnv` and a clean renderer. Every
  extra feature competes with that clarity.
- **Alternatives:** a feature-rich environment.
- **Trade-off:** a deliberately limited scope (D-03). It is also an excellent size for learning.

### D-03: Explicit non-goals
- **Status:** Decided. Out of scope: deadlock detection, confidence intervals, CI-grade evaluation,
  multiple layouts, fixed-task mode, a centralised planner, a yielding baseline, any claim about
  generalisation, re-implementing PPO, touching `warehouse_rl/`. (Confidence intervals were moved
  into scope on 2026-11-03, see D-35.)
- **Why:** Writing the non-goals down stops scope creep and tells reviewers what *not* to expect.
- **Trade-off:** results are weak scientifically (see D-35). They are good enough for a demo.

### D-04: Five stages, each ending in something runnable
- **Status:** Decided. (1) rules and environment; (2) renderer; (3) policies, evaluation, demo;
  (4) ship the scripted version; (5) RL (stretch).
- **Why:** Every stage produces a demonstrable result, and Stage 4 is a releasable stopping point,
  so the project is "finished" before the riskiest part (RL dependencies) begins.
- **Alternatives:** build everything, integrate at the end.
- **Trade-off:** slightly more up-front structure; much lower risk.

### D-05: Anti-goals as guard rails
- **Status:** Decided. (a) No stage that cannot be run. (b) A demo that never crashes just because
  an optional piece (the RL model file) is missing. (c) Never report an RL result that was not
  measured, and never claim it beats greedy unless it does.
- **Why:** These are the failure modes that make a showcase project embarrassing. They are turned
  into tests where possible (D-37, D-44).

---

## B. Environment: world and interface

### D-06: PettingZoo `ParallelEnv`, not AEC
- **Status:** Decided.
- **Why:** All robots move simultaneously, which is exactly the Parallel API's model
  (`step({agent: action})` -> five dicts). AEC would serialise the moves and hide the design
  problem the project is actually about: resolving simultaneous conflicts (D-14). The Parallel API
  also maps directly onto SuperSuit's vectorisation (D-40). *(partly inferred)*
- **Alternatives:** AEC (turn-based); a plain Gymnasium env with a joint action.
- **Trade-off:** the environment itself must define what happens on conflicts; in return the
  semantics are honest.
- **Verification:** `pettingzoo.test.parallel_api_test(MultiAgentWarehouseEnv(), num_cycles=300)` is a
  required test. 300 is larger than `max_steps = 200`, so a full episode, including truncation, is
  exercised (the test also resets twice). *(inferred from reading the installed test)*

### D-07: One fixed 8 x 8 layout in a settings file
- **Status:** Decided. Four 2 x 2 shelf blocks, `.` free and `#` shelf (picture in the study notes).
  The map and every game setting (robot counts, step limit, reward values, render fps) live in
  `multi_agent_warehouse/config.py`; `env.py` imports them and defines none.
- **Why:** One map keeps every rule testable and the rendering numbers fixed (a 512 x 552 frame).
  The shelf blocks leave aisles, so robots *must* meet and negotiate. Keeping the settings in one
  importable file means changing the map or a reward is a one-file edit, and the grid size is read
  from `LAYOUT` rather than assumed. `tests/test_config.py` guards the map (8 x 8, only `.` and `#`,
  no sealed-off cells).
- **Alternatives:** several layouts, procedural generation, a YAML or JSON file (needs a parser or
  package data and fails later than an import error), an `env` keyword for the layout.
- **Trade-off:** no generalisation claim is possible (explicit non-goal, D-03, and the README must
  say so).

### D-08: Robot count and naming
- **Status:** Decided. `n_robots` is 1 to 6, default 3; ids `robot_0 ... robot_{n-1}`; any other
  count raises `ValueError` (D-24).
- **Why:** Three robots is enough for interesting interactions and a readable picture; six is
  the number of colours the renderer defines (D-31). String names are required by the
  dict-keyed PettingZoo API.

### D-09: Starts and targets
- **Status:** Decided. Robots start on **distinct random free cells**. Each has one target: a random
  free cell different from its own cell. Targets of different robots may coincide. After a delivery
  the robot immediately gets a new target (a free cell different from where it is standing).
- **Why:** Distinct starts are required by the "no sharing a cell" rule. A target different from
  the start avoids an instant free delivery. Allowing coincident targets removes a needless
  constraint and creates natural contention. Immediate re-targeting makes the task continuous, so
  there is no "done" state and no terminations (D-22).
- **Alternatives:** a fixed task list; a pickup-and-drop-off cycle; shelves as pickup points.
- **Trade-off:** it is a navigation task, not a full logistics problem. "Warehouse" is the theme,
  not the model.

### D-10: Five discrete actions
- **Status:** Decided. `Discrete(5)`: 0 stay, 1 up (row-1), 2 down (row+1), 3 left (col-1),
  4 right (col+1). Same convention as `warehouse_rl`.
- **Why:** The smallest action set that can reach every cell, with an explicit "stay" so waiting is
  a real choice (important for yielding). Matching `warehouse_rl` keeps the two projects comparable.
- **Alternatives:** four moves with no stay; diagonals; continuous velocity.

### D-11: An 8-number local observation
- **Status:** Decided (confirmed 2026-11-03). `Box(-1, 1, shape=(8,), float32)`: own row and column, target offset
  (rows, columns), and four neighbour-blocked flags (up, down, left, right). `warehouse_rl` used
  5 x 5 local windows instead.
- **Why:** It is small enough to read and to hand-check (worked example in the study notes), tiny
  for an MLP to learn from, and the target is **egocentric** (an offset, not absolute coordinates).
  It is deliberately *partial*: a robot cannot see other robots' targets, nor anyone beyond adjacent
  cells. That keeps the problem genuinely multi-agent, because the robot must infer from neighbour
  flags that someone is in the way.
- **Alternatives:** 5 x 5 window (more information, 25+ values per channel); full global state;
  one-hot grids.
- **Trade-off:** less information means a lower ceiling for RL (a robot cannot plan around a jam it
  cannot see), but a much smaller and faster problem.

### D-12: Normalise by `max(H, W) - 1`
- **Status:** Decided. Row, column and both offsets are divided by `max(H, W) - 1` (7 on the
  8 x 8 map, stored as `_scale` in `env.py`) so all values lie in `[-1, 1]`.
- **Why:** Neural networks train far better on small, centred inputs; it also makes the declared
  `Box(-1, 1)` bounds truthful. Using `max(H, W) - 1` (not each axis separately) preserves the grid's
  aspect ratio and would still be valid if the grid became non-square.

### D-13: Neighbour flags include other robots
- **Status:** Decided. A flag is 1.0 if the neighbour is out of bounds, a shelf, **or occupied by
  another robot**.
- **Why:** The most useful local signal for avoiding a collision is "someone is right there".
  Merging all three causes into one flag keeps the vector at 8 numbers and lets `GreedyPolicy`
  treat "blocked" uniformly (D-34).
- **Trade-off:** a robot cannot tell a wall from a robot. A moving robot might clear next step, a
  shelf never will. The agent cannot learn that distinction from this observation.

---

## C. Physics and rewards

### D-14: Move resolution as two rules plus a fixed point
- **Status:** Decided (confirmed 2026-11-03). A pure function `resolve_moves(positions, actions, shelves)` in
  `rules.py` returns new positions and the set of collided ids; inputs are not mutated.
  1. **Same cell:** a robot moving into a cell another robot also proposes is blocked. A stayer
     (or a robot bumping a wall) proposes its own cell, so moving into a stayer is the same rule.
  2. **Swap:** two robots exchanging cells are both blocked.
  Blocked robots stay, then count as stayers, and the rules repeat until nothing changes.
- **Why:** Simultaneous movement needs *some* conflict rule, and a small rule set can be tested
  exhaustively (the plan's 11-case table). Making it a pure function separates physics from
  PettingZoo plumbing, so it is testable without building an environment.
- **Alternatives:** random priority order; fixed priority per robot id (unfair); queue-based
  reservation; processing robots sequentially (then robot_0 always wins).
- **Trade-off:** conflicts always cancel *everyone* involved; nobody wins. That is symmetric and
  fair, but it means two robots can block each other repeatedly, and the policy must learn to
  break the tie.
- **Implementation note (learned the hard way):** each round must decide *all* blocks from one
  snapshot of the proposals, then apply them together. Updating proposals while still checking
  other robots makes the result depend on iteration order (a prototype of this failed 3 of the
  plan's 11 tests that way).

### D-15: Four-robot rotations are allowed
- **Status:** Decided (confirmed 2026-11-03); described as a *deliberate simplification* versus `warehouse_rl`.
- **Why:** Four robots circling a 2 x 2 block never propose the same cell and never form a pairwise
  swap, so the two rules do not catch it. Catching it would need cycle detection for little
  benefit.
- **Alternatives:** detect cycles and block them (more code, closer to physical reality).
- **Trade-off:** slightly unrealistic behaviour that is explicit and tested
  (`test_rotation_is_allowed`).

### D-16: Robots held up behind a conflict are penalised too
- **Status:** Decided (confirmed 2026-11-03; revisit if RL learns oddly cautious behaviour). A robot that tried to move and was blocked, directly or through the
  repeat rule, collides. `warehouse_rl` exempts those held-up robots.
- **Why:** Simpler to define and to test: "tried to move, ended up blocked by another robot".
- **Trade-off:** a robot can be penalised for another pair's conflict, which weakens credit
  assignment for RL (it may learn to avoid queues, not just conflicts). Greedy is unaffected.

### D-17: Bumping a wall or shelf is not a collision
- **Status:** Decided. The robot stays; it pays only the step cost. A robot that chose to stay
  never collides.
- **Why:** The "collision" metric and penalty are about robot-robot interference, which is the
  multi-agent part. Penalising walls would blend in a navigation signal and make the collision
  count mean two things. (`test_wall_and_shelf_bump_are_not_collisions`.)

### D-18: Step order and observation last
- **Status:** Decided. Validate actions -> resolve moves -> check targets -> compute rewards ->
  advance the clock -> build observations.
- **Why:** A fixed order makes the semantics unambiguous. Building observations last means a robot
  that just delivered already *sees its new target* in the observation it acts on next; otherwise
  it would act on a stale target for one step.

### D-19: Individual rewards
- **Status:** Decided. Each robot receives its own delivery bonus and collision penalty.
- **Why:** Clear credit assignment makes learning easier (see [study note 02](study-notes/02-multi-agent-systems.md)), and it is the natural
  fit for parameter sharing with identical robots.
- **Alternatives:** one shared team reward (aligned but noisy); mixed.
- **Trade-off:** robots are not directly rewarded for helping each other; cooperation emerges only
  through avoiding penalties.

### D-20: Reward magnitudes
- **Status:** Decided. Delivery +1.0; every step -0.01 for every robot; collision -0.1 per collided
  robot.
- **Why:** Delivery dominates (worth 100 idle steps), so it is the clear goal. The per-step cost
  nudges speed without overwhelming it (over 200 steps the worst case is -2.0). A collision costs
  10 steps' worth, large enough to be noticed, small enough that a robot will still risk one for a
  delivery. *(partly inferred)*

### D-21: Potential-based shaping, off by default, never used in evaluation
- **Status:** Decided. Optional bonus `+0.02 x (Manhattan distance to target before the step -
  after)`, measured against the target held at the start of the step. Off by default; on during
  training only.
- **Why:** The +1 delivery reward is sparse, and a random-walking robot rarely scores, so RL
  learns slowly without a hint. A potential-based shape is the principled way to add a hint
  (Ng et al., 1999). Keeping it **out of evaluation** means reported numbers measure the real task
  (D-35).
- **Alternatives:** no shaping (slow or failed learning); a larger coefficient (risks teaching
  "approach" over "deliver"); curriculum learning.
- **Trade-off / caveat:** the optimal-policy guarantee is exact only for `gamma = 1` and within one
  target's lifetime. PPO's default `gamma` is 0.99, and a delivery changes the target, which makes
  the sum of shaping per target equal `0.02 x its starting distance` (at most 0.28 on the 8 x 8
  map, where the largest Manhattan distance is 14), a small distance-proportional bonus per
  delivery. The spec's wording ("should not change which policy is best") is therefore
  *approximately* true. Plan Task 8 keeps the coefficient adjustable if training needs it.
  Manhattan distance is also only a sound potential while shelves force few detours: on the
  current map the shortest path is never more than 2 steps longer, but a map with dead ends would
  make it punish correct moves (see
  [study note 05](study-notes/05-reward-shaping.md) and [14](study-notes/14-generalization.md)).

---

## D. Episode, state and robustness

### D-22: No terminations; truncate everyone together
- **Status:** Decided. All robots are truncated together when `step_count` reaches `max_steps`
  (default 200); afterwards `env.agents == []`.
- **Why:** A warehouse never finishes, so there is nothing to "terminate". A time limit is by
  definition a truncation, which also tells RL algorithms to keep bootstrapping value at the cut-off.
  Ending all agents at the same instant keeps the agent set constant during an episode, a requirement
  of the SuperSuit vectorisation used for RL (D-40). It also makes evaluation simple: every
  episode is the same length.
- **Alternatives:** terminate a robot after N deliveries (varying episode lengths, harder to
  vectorise).

### D-23: `step()` after the episode ended returns empty dicts
- **Status:** Implemented (Review Focus 1). `step()` with `env.agents == []` returns five empty dicts and
  does not raise.
- **Why:** A forgotten `while env.agents` check would otherwise crash a long training run.
  *(inferred)* PettingZoo does not forbid this behaviour.
- **Alternatives:** raise an error; ignore silently by returning stale data.
- **Trade-off:** hides a caller bug rather than surfacing it. A test pins the behaviour so it is at
  least deliberate.

### D-24: Validate inputs before changing state
- **Status:** Implemented (Review Focus 2 and 3). A missing key, an extra key, an action outside 0 to 4, or
  a non-integer in `step`, and `n_robots` outside 1 to 6 or an unknown `render_mode` in the
  constructor, all raise `ValueError` *before* any state changes.
- **Why:** Fail loudly and early with a message naming the problem. Raising before mutating means a
  bad call cannot leave the environment half-updated.

### D-25: One seeded generator, no global randomness
- **Status:** Decided. `reset(seed=s)` builds a single `np.random.default_rng(s)` that drives starts
  and targets and nothing else; `reset()` without a seed continues the stream; rendering never
  consumes it. No global RNG anywhere.
- **Why:** Reproducibility is a precondition for any measurement. Same seed, same episode, which
  also lets evaluation compare policies on identical episodes. Keeping render out of the stream means
  watching the demo cannot change the result (`test_rendering_does_not_change_trajectory`).
- **Alternatives:** `np.random.seed` / `random` (global, easy to disturb from elsewhere).

### D-26: Public state attributes and `info` keys
- **Status:** Decided. Public: `positions`, `targets` (`dict[str, tuple[int, int]]`), `shelves`
  (bool array), `step_count`, `total_deliveries`, `total_collisions`, `hud_note`.
  `infos[agent]` carries `deliveries` (running count) and `collided` (bool, this step).
- **Why:** The renderer, tests and demo all need to read state, and the tests need to set it
  (for example placing a robot next to its target to test a delivery). Plain attributes are the
  simplest way to allow both. `hud_note` lets `play.py` show the policy name and "PAUSED" without
  the renderer knowing about policies.
- **Trade-off:** anyone can mutate state from outside. Policies promise (D-33) not to read it.

### D-27: Cache the spaces with `lru_cache`
- **Status:** Implemented. `observation_space(agent)` and `action_space(agent)` are decorated
  with `functools.cache` in `env.py` (the same as `lru_cache(maxsize=None)`, which the plan names),
  with `# noqa: B019` because caching a method keeps `self` alive.
- **Why:** PettingZoo's API test asserts that the same call returns the **identical object**
  (`is`, not `==`), because spaces hold their own seedable RNG state. Verified in the installed
  `parallel_test.py`.

### D-28: Iterate robots in sorted-id order
- **Status:** Implemented. `resolve_moves`, policies and batch building iterate ids in sorted order.
- **Why:** Dict order is insertion order, which can vary. Fixed order gives deterministic results
  and consistent batching for `RLPolicy` (D-40). With the snapshot rule of D-14 the physics no
  longer depends on order, so this is defence in depth there and essential for the RNG draws.

---

## E. Rendering

### D-29: Draw offscreen first
- **Status:** Decided. `Renderer` always draws to an in-memory `pygame.Surface` and returns an
  `(H_px, W_px, 3)` uint8 array. `human` mode additionally blits to a window at `render_fps` (10).
- **Why:** One drawing path serves the window, the GIF recorder and the tests; the pixel-level
  tests can run with no display.
- **Alternatives:** separate code paths for window and array (can drift apart); drawing directly to
  the display (cannot be tested headless).

### D-30: Import Pygame lazily, and only in the renderer
- **Status:** Decided. `pygame` is imported inside `render.py` only; the renderer is created on the
  first `render()` call.
- **Why:** A user who only wants the environment (for training, say) should not need a display
  library to import it. `render_mode=None` truly costs nothing.

### D-31: Fixed visual constants
- **Status:** Decided. 40 px HUD bar on top; grid at 64 px per cell; floor light grey; shelves dark
  brown; each robot a filled circle in its own colour (six distinct colours); each target an
  outlined square inset 8 px in the same colour (line width 6); `Font(None, 24)` HUD text.
- **Why:** Numbers fixed in the spec let tests assert exact pixels (for example the centre of a
  robot's cell equals `ROBOT_COLORS[0]`). Matching colours link each robot to its target. The
  HUD shows step, deliveries, collisions and a free-text note.
- **Resulting frame size:** `(40 + 8*64, 8*64, 3) = (552, 512, 3)`.

### D-32: `render_mode=None` means do nothing
- **Status:** Decided. `render()` returns `None` and opens no window. `close()` is safe to call
  twice.
- **Why:** Follows the PettingZoo / Gymnasium convention and makes training headless by default.

---

## F. Policies, evaluation and demo

### D-33: A two-method policy interface, observations only
- **Status:** Decided. `Policy` is a `typing.Protocol` with `reset(seed)` and
  `act(observations) -> dict[str, int]`. Policies use only observations, never env internals.
- **Why:** One interface lets the demo, the evaluation and the RL agent share a loop. Observation-only
  access keeps comparisons fair: greedy gets no information that a learned policy would not.
  `Protocol` (structural typing) avoids a base class for something this small. *(partly inferred)*

### D-34: The greedy rule with a random sidestep
- **Status:** Decided. Step along the axis with the larger absolute offset (ties go to rows). If that
  is blocked, try the other axis if its offset is non-zero. If still blocked, or the other offset is
  zero, take a random unblocked direction (own RNG). If none, or on target, stay.
- **Why:** A pure "move toward the target" rule gets stuck against any shelf. The random
  sidestep is the cheapest fix that stays within the observation. It also shows the robots do
  not need to know the map.
- **Alternatives:** A* pathfinding (needs the map, which violates D-33); wall-following.
- **Trade-off:** random sidestepping is inefficient and can wander. It is a baseline, not a solver.
  The plan requires it to deliver at least 2x random's deliveries, and says to fix the rule, not
  the threshold, if it does not.

### D-35: A tiny, fair evaluation
- **Status:** Decided; revised 2026-11-03. `python -m multi_agent_warehouse.evaluate` runs each
  available policy for 100 episodes by default (seeds 1000 to 1099, 3 robots, 200 steps, shaping
  off; `--episodes N` changes it) and prints mean deliveries and mean collisions per episode, each
  with a 95% confidence half-width (1.96 standard errors). The original plan used 20 episodes and no
  intervals. No deadlock flag.
- **Why:** The goal is that "any claim in the README has a number behind it". Using the **same
  seeds** for every policy makes the comparison paired. Shaping off means the reported reward
  reflects the true task.
- **Alternatives:** 20 episodes and no intervals (the original); a paired test between policies.
- **Trade-off:** the intervals use a normal approximation and treat each policy separately, so
  overlapping intervals mean "cannot tell apart", not "equal". Current numbers: random 2.63 ± 0.34
  deliveries, greedy 12.67 ± 2.07.

### D-36: Entry points and demo controls
- **Status:** Decided (built and merged). `play.py` is in the **repo root** (the file a reviewer runs).
  Training and evaluation are modules (`python -m multi_agent_warehouse.train`,
  `python -m multi_agent_warehouse.evaluate`) so tests can import them. Keys: space pause, R reset
  (same seed + 1), P cycle policy, Esc or Q quit. Flags: `--policy {random,greedy,rl}` (default
  greedy), `--robots` (3), `--seed` (0), `--fps` (10), `--frames N` (0 = until quit). A finished
  episode auto-resets with the next seed.
- **Why:** `--frames` exists so the demo can be smoke-tested headlessly in CI. Auto-reset keeps the
  demo looping without a keypress. `argparse` `choices` rejects unknown policies for free.

### D-37: The demo degrades gracefully
- **Status:** Decided (Review Focus 5). With the RL model file missing, `play.py` prints a one-line
  message naming `rl` and continues with random and greedy; `--policy rl` falls back to greedy.
  `available_policies()` offers `rl` only if the model file exists and Stable-Baselines3 imports.
- **Why:** Anti-goal D-05(b). A clone must always run, with or without the optional extras.

### D-38: No keyboard-controlled robot yet
- **Status:** Unconfirmed. Left out of the plan; can be added after Stage 4.
- **Why:** It would be fun but adds input handling and a mixed human/AI loop without helping the
  core goal.

---

## G. Reinforcement learning (stretch)

### D-39: Use Stable-Baselines3 PPO; do not reimplement it
- **Status:** Decided. `PPO("MlpPolicy", ...)` from Stable-Baselines3.
- **Why:** The point is to demonstrate the environment, not to write an optimiser. PPO is stable,
  widely used, and works for discrete actions. `MlpPolicy` is the right size for an 8-number
  observation.
- **Fallback if the stack does not work:** a hand-written PyTorch PPO, treated as separate work (D-42).

### D-40: Parameter sharing through SuperSuit
- **Status:** Decided. `pettingzoo_env_to_vec_env_v1` turns each robot into one sub-environment of a
  vectorised env; `concat_vec_envs_v1(vec, n_envs, num_cpus=1, base_class="stable_baselines3")`
  replicates it. One network is trained on all robots' experience. At run time `RLPolicy`
  stacks observations (sorted-id order), calls `predict` once, and maps results back.
- **Why:** Robots are identical, so one shared policy learns n times faster and generalises across
  robot ids. It is also the simplest multi-agent training that works with single-agent libraries.
- **Alternatives:** independent networks per robot; CTDE with a central critic (MAPPO/QMIX; more
  machinery, not needed here).
- **Trade-off:** the robots cannot specialise, and identical inputs give identical outputs, hence D-43.
  It requires identical spaces for all agents and a fixed agent set (satisfied by D-22).

### D-41: RL dependencies are an optional extra, imported lazily
- **Status:** Decided and applied (Task 7, 2026-11-07). `pyproject.toml` declares
  `rl = ["stable-baselines3>=2.3", "supersuit>=3.9", "torch>=2.2"]`. SB3 and SuperSuit are imported inside
  `train()` and `RLPolicy.__init__`. The RL test uses `pytest.importorskip`.
- **Why:** PyTorch is heavy. The scripted version, which is the releasable core, must install and
  run without it.

### D-42: A spike with a stop-and-ask gate
- **Status:** Decided. The first RL task installs the extra and runs 2,048 training steps. If that
  fails and the fix is not a small version change, work stops and asks.
- **Why:** SuperSuit / SB3 / PettingZoo / Gymnasium version compatibility is the biggest unknown in
  the project, and it is cheaper to find out in a 10-minute spike than after writing the RL
  code. The decision to switch to a custom PPO is the user's, because it is a different scale of
  work.
- **Outcome (2026-11-07): passed, with one code workaround and no version change.** `pip install -e ".[rl]"`
  resolved to SB3 2.9.2, SuperSuit 3.11.0 and PyTorch 2.15.0 alongside the existing PettingZoo 1.27.0,
  Gymnasium 1.4.0 and NumPy 2.4.6 (`pip check` clean). 2,048 steps train and save a loadable model.
- **One incompatibility:** `PPO(..., seed=seed)` crashes with `'ConcatVecEnv' object has no attribute
  'seed'`, because SB3 calls `vec_env.seed()` and SuperSuit's `ConcatVecEnv` does not define it
  (training with `seed=None` works). A version pin does not help. `train()` seeds outside PPO instead:
  SB3's `set_random_seed(seed)` for Python, NumPy and torch, plus `vec_env.venv.reset(seed=seed)`, which
  `ConcatVecEnv` supports. Same seed gives identical weights; a different seed gives different weights.
  This is a code change rather than a version change, so strictly it is outside the gate's wording; it
  was judged small enough not to need the custom-PPO decision.
- **Harmless noise:** two PettingZoo deprecation warnings, raised inside SuperSuit's imports.

### D-43: Sample actions at inference time
- **Status:** Decided. `RLPolicy.act` uses `model.predict(batch, deterministic=False)`.
- **Why:** Shared weights plus similar observations can make robots mirror each other into a
  head-on standoff forever. Sampling injects the randomness that breaks the symmetry, the same
  role the random sidestep plays for greedy (D-34).
- **Trade-off:** run-to-run variability (mitigated by seeded evaluation).

### D-44: Commit the model and report honestly
- **Status:** Decided. Train once with shaping on; commit `models/ppo_multi_agent_warehouse.zip` (expected
  under 1 MB, checked under 2 MB). **Acceptance bar:** mean deliveries at least 2x random's.
  Beating greedy is *not* required. If missed: more steps (1,000,000), then a different shaping
  coefficient. The README records the settings and says plainly where RL lands, plus what it was not
  tested on (one layout, 3 robots, no generalisation).
- **Outcome (Task 8, 2026-11-07):** Met on the first run, so no retraining. Defaults (300,000 steps, 8 envs,
  seed 0, `SHAPING_SCALE` 0.02) gave 51.07 deliveries/episode vs random 2.63 and greedy 12.67, with 189.95
  collisions vs greedy's 401.30 (100 seeds, 1000 to 1099, 95% CIs in the README). Model is 146 KB.
- **Why:** A clone should be able to play the trained agent without training. Fixing the bar before
  seeing results prevents moving the goalposts (anti-goal D-05(c)).

### D-52: Set PPO's rollout size and training budget explicitly
- **Status:** Decided (accepted by you, 2026-10-28) and **settled in Task 7 (2026-11-07)**: `n_steps = 128`,
  `batch_size = 256`, `total_steps = 300_000`, `n_envs = 8` (the `train()` defaults). Not in the spec or
  plan, so neither document mentions it; raised while writing
  [study note 04](study-notes/04-ppo.md#a-budget-problem-the-plan-does-not-yet-mention-computed).
  The Context figures below are **calculated** from the plan's numbers and SB3's defaults (since
  confirmed against the installed SB3); the measurements are in the Outcome entry.
- **Context:** Task 7 trains `PPO("MlpPolicy", vec_env, seed=seed, verbose=0)` with `n_envs = 8` and
  `total_steps = 300_000`, leaving `n_steps` at its default of 2048 *per sub-environment*. SuperSuit
  makes each robot its own sub-environment (D-40), so there are `3 x 8 = 24`. One rollout is
  `2048 x 24 = 49,152` samples, and 300,000 steps is only `ceil(300,000 / 49,152) = 7` policy
  updates from about 500 real episodes (`300,000 / 3 robots / 200 steps`).
- **Outcome (measured 2026-11-07, 300,000 steps, 30 evaluation episodes from seed 1000, shaping off):**
  random 2.4 deliveries per episode, greedy 13.6.

  | `n_steps` / `batch_size` | Deliveries per episode | Collisions per episode |
  | --- | --- | --- |
  | 2048 / 64 (SB3 defaults), seed 0 | 41.4 | 35.7 |
  | 256 / 256, seeds 0, 1, 2 | 50.8, 57.2, 49.8 | 189.6, 126.1, 189.6 |
  | 128 / 256, seeds 0, 1, 2 | 54.7, 58.5, 60.1 | 170.9, 136.3, 136.9 |

  Every setting clears the D-44 bar (2x random) by a wide margin and beats greedy, so the budget was not
  the bottleneck. 128 / 256 was chosen for the highest and steadiest deliveries (about 14 s per run).
  Caveats: three seeds at most; the default 2048 setting collides far less (36 versus about 137 to 171) but
  delivers less; the evaluation is 30 episodes, so differences of a few deliveries are within noise.
  Also, a request for 2,048 steps runs whole rollouts (12,288 timesteps with 2 copies), so
  `total_steps` is a lower bound.
- **Proposal (as originally made):** Settle these in Task 7's spike and record the final values in the README (D-44
  already requires the settings to be recorded):
  - `n_steps`: 128 gives 98 updates and 256 gives 49 for the same 300,000 steps;
  - `total_steps` and `batch_size`, chosen together with `n_steps`;
  - `verbose=1` while tuning, so the training log can be read;
  - judge progress with `evaluate.py` (shaping off, fixed seeds), never the training reward.
- **Alternatives:**
  - keep the defaults and train longer: 1,000,000 steps gives about 21 updates, and it is the
    plan's own fallback;
  - fewer copies: `n_envs = 2` is 6 sub-environments and 12,288 samples per rollout, about 25
    updates for 300,000 steps, but slower data collection;
  - leave it and see what the spike does (cheap, but a wasted run if it fails for this reason).
- **Trade-off:** a smaller `n_steps` gives more updates from the same data, at the cost of noisier
  advantage estimates and rollouts that end mid-episode (episodes are 200 steps, so a 128-step
  rollout is shorter than one). PPO handles that by bootstrapping from the critic. The acceptance
  bar in D-44 does not change.

### D-53: Generalization is a separate Stage 6, built one level at a time
- **Status:** Decided (accepted by you, 2026-10-28). Not in the spec or plan, so Stage 6 is not in
  either yet. Today's behaviour is a stated non-goal (D-03:
  "any claim about generalization"; D-07: one fixed layout), so this would revise decisions, not
  fill a gap.
- **Why:** Trained weights are tied to one layout. They load on any map (the model maps 8 numbers
  to 5 actions) and then degrade **silently**. The causes are the absolute row and column inputs,
  an offset scaled by the grid size, a single training map, a policy with no memory, and a view that
  stops at adjacent cells ([study note 14](study-notes/14-generalization.md)).
- **Proposal:** After Stage 5, one level at a time, measuring after each:
  0. held-out evaluation, to find out how bad it is;
  1. remove map-identifying inputs (absolute position; a grid-size-dependent scale);
  2. train on many layouts, chosen per episode (for example through `reset(options=...)`), with
     a connectivity check, a structured generator and a fixed observation shape;
  3. path guidance (a breadth-first distance field) with path-distance shaping and a matching
     "BFS greedy" baseline;
  4. a wider window or memory, only if needed;
  5. separate train, validation and test layouts, reported separately.
- **Gate:** after levels 1 and 2, does RL on held-out maps still beat random by the same 2x margin as
  D-44, measured on those same maps? If not, add level 3 before anything fancier.
- **Why not now:** a measured single-layout baseline is needed first to compare against. The riskiest
  unknown, the SuperSuit and SB3 stack (D-42), should be answered before more is built on it. And a
  layout-generating, map-aware version would not fit the project's "about 450 lines, readable in one
  sitting" goal (D-02).
- **Alternatives:** fold it into Stage 5 (it would confound the dependency spike and the baseline);
  a separate package, as D-01 did for the mini version; or never, and let the README state the
  limitation.
- **Trade-off:** level 3 assumes robots know the map, which is realistic for a warehouse (a route
  planner plus local avoidance) but is not learning navigation from scratch. The code and test
  surface grows, and the following decisions would be revised:

  | Decision | Today | Under Stage 6 |
  | --- | --- | --- |
  | D-03, D-07 | One layout, no generalisation claim | Several layouts, a measured claim |
  | D-11 | 8 numbers including absolute position | Fewer map-identifying inputs, optional guidance features |
  | D-12 | Divide by `max(H, W) - 1` | A fixed constant, clipped |
  | D-21 | Manhattan shaping | Path-distance shaping |
  | D-35, D-44 | One-layout evaluation and bar | Held-out layout evaluation and bar |

---

## H. Tooling and process

### D-45: Python 3.10+ with the project venv
- **Status:** Decided. `requires-python = ">=3.10"`; the `.venv` is Python 3.11; always run tools as
  `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python`.
- **Why:** 3.10 gives modern typing syntax (`int | None`, `list[str]`); the venv pins what the
  tests actually ran on. Core dependencies are only NumPy, Gymnasium, PettingZoo and Pygame.

### D-46: Ruff lint and format exactly as CI does
- **Status:** Decided. Rules `E, F, I, B, UP`, line length 100, and `ruff format --check`.
- **Why:** "Passes locally" must mean "passes CI". The rule sets cover errors, unused code, import
  order, likely bugs and modern-syntax upgrades.

### D-47: Test first, flat test layout
- **Status:** Implemented. Each task starts with failing tests, then the implementation. Tests live flat in
  `tests/` named `test_*.py` with no `__init__.py`, so basenames must be unique.
- **Why:** Test-first proves each test can fail, and the plan's tables spell out the exact
  assertions. Unique basenames matter because pytest's default import mode needs them without
  `__init__.py`.

### D-48: Headless testing with `SDL_VIDEODRIVER=dummy`
- **Status:** Implemented. Any test file that imports `pygame` or the renderer sets
  `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` at the top; `test_play.py` passes it to the
  subprocess.
- **Why:** Tests must pass on machines (and CI) with no display. `setdefault` lets a developer
  override it.

### D-49: Type hints and docstrings on the public surface
- **Status:** Implemented. Type hints on public functions; a docstring on every module and class.
  Test files may omit them. No pandas, no SciPy.
- **Why:** The project is meant to be read; types and docstrings are the cheapest documentation.
  Avoiding heavy dependencies keeps installation light.

### D-50: Pillow for the README GIF
- **Status:** Decided (confirmed 2026-11-03). `scripts/record_demo.py` writes
  `docs/multi_agent_warehouse/demo.gif` (greedy, seed 0, 100 steps, `duration=100`, `loop=0`; target under 3 MB).
- **Why:** A GIF shows the project working in the README without anyone running it. Pillow is the
  lightest way to write one.
- **Current state:** `pyproject.toml` lists `pillow` in the `dev` extra (the spec's assumption that
  matplotlib installs it no longer holds). `scripts/record_demo.py` is kept local only (excluded
  through `.git/info/exclude`), as is this `docs/` folder.

### D-51: Branch and commit conventions
- **Status:** Implemented, with a changed branch scheme. Commit messages are `type(scope): message`
  (for example `feat(warehouse): add move resolution`). The plan's single `multi-agent-warehouse`
  branch became one branch and PR per task or chore (`feat/render-pygame`, `chores/readme-rewrite`,
  and so on), all merged into `main`.

---

## I. Open questions, risks and inconsistencies

Found while writing these notes; updated 2026-11-03. Struck-through items are resolved.

| # | Item | Why it matters | Suggested action |
| --- | --- | --- | --- |
| 1 | **`.gitignore` contains `docs/`** | The spec, plan, these notes and the study notes are not in git, so they exist only on this machine. Only `demo.gif` is tracked (force-added). `scripts/record_demo.py` is also kept local (`.git/info/exclude`). | **You chose to keep docs and the script local (2026-11-03).** The public README no longer refers to the notes. |
| 2 | ~~Unconfirmed decisions~~ | Resolved 2026-11-03: D-11, D-14, D-15, D-16 and D-50 confirmed as they are; D-01 and D-36 are built and merged. D-38 (no keyboard-controlled robot) is still unbuilt. | Revisit D-16 if RL learns oddly cautious behaviour. |
| 3 | ~~`warehouse_rl/`, `DESIGN.md`, branch `mvp-build` not in this folder~~ | Resolved 2026-11-03: they are not part of this repository. The spec, the plan and these notes still mention them as background; read those mentions as historical. | None. |
| 4 | ~~Repo has no commits~~ | Resolved: the repo now has commits for the scaffold, Task 1, the rename and Task 2. | None. |
| 5 | ~~Spec assumption on Pillow is stale~~ | Resolved: Pillow is a declared `dev` dependency (D-50). The spec's matplotlib sentence is annotated as stale. | None. |
| 6 | ~~No `rl` extra in `pyproject.toml` yet~~ | Resolved 2026-11-07: the `rl` extra is declared (Task 7). | None. |
| 7 | ~~SuperSuit compatibility unverified~~ | Resolved 2026-11-07 (D-42): the stack installs and trains; `PPO(seed=...)` needs the workaround described in D-42. | Re-run the spike if any of SB3, SuperSuit, PettingZoo or Gymnasium is upgraded. |
| 8 | **Shaping is only approximately policy-invariant** | See D-21. | Fine for this project; know it before quoting "does not change the optimal policy". |
| 9 | **Collision penalty on held-up robots** (D-16) | Can muddy credit assignment for RL. | Revisit if RL learns oddly cautious behaviour. |
| 10 | ~~Evaluation has 20 episodes and no intervals~~ | Resolved 2026-11-03 (D-35): 100 episodes by default with 95% confidence half-widths and an `--episodes` flag. | A close PPO-versus-greedy result should still be checked with more episodes. |
| 11 | **`step()` after the end returns empty dicts** (D-23) | Hides caller bugs. | Keep unless it hides a real bug during RL work. |
| 12 | ~~PPO's training budget looks too small~~ (D-39, D-44; [D-52](#d-52-set-ppos-rollout-size-and-training-budget-explicitly)) | Resolved 2026-11-07: measured, and `n_steps = 128`, `batch_size = 256` chosen. Every tried setting beat the D-44 bar, so the default was not a blocker. See [study note 04](study-notes/04-ppo.md#a-budget-problem-the-plan-does-not-yet-mention-computed). | Judge the final model with `evaluate.py` (more than 30 episodes), never the training reward. |
| 13 | **Trained weights are tied to one layout** (D-03, D-07, D-11; decided in [D-53](#d-53-generalization-is-a-separate-stage-6-built-one-level-at-a-time)) | The model loads on any map and degrades silently. See [study note 14](study-notes/14-generalization.md). | Treat generalization as a separate Stage 6 after Stage 5, not as part of it. |

## Changing a decision

Append a new entry or edit the status line; do not rewrite history silently. For each change record:
what changed, why, which tests changed, and whether the study notes need updating.
