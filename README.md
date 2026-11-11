<h1 align="center">Multi-Agent Warehouse</h1>
<p align="center"><b>Three robots, one warehouse, one shared recurrent PPO brain. A from-scratch multi-agent RL project: the environment, the baselines, the training pipeline, and a set of seeded experiments that show what actually improved the agent.</b></p>

<p align="center">
  <img src="docs/evidence/demo.gif" alt="The tuned recurrent PPO agent controlling three robots that deliver items in the warehouse" width="420"><br>
  <sub>The shipped agent (tuned recurrent PPO, 1M steps) over 100 steps on evaluation seed 48: 43 deliveries, 6 collisions. Seed 48 is a clean example picked from seeds 40 to 79; the average over those seeds is about 31 deliveries and 25 collisions per 100 steps. All three robots run the same network.</sub>
</p>

---

## At a Glance

- **The task:** three robots move simultaneously on an 8 x 8 grid. Each one reaches its target item, is
  handed a new target on delivery, and is penalised for bumping into other robots.
- **The learner:** one recurrent PPO network (Stable-Baselines3 / sb3-contrib, an LSTM policy) shared
  by all robots through SuperSuit parameter sharing, trained for 1,000,000 steps (about 35 minutes
  on a laptop CPU).
- **The result:** **62.0 deliveries per episode, against 12.7 for a hand-written greedy rule and 2.6
  for random**, with about an eighth of greedy's collisions (51 against 401). Measured over 100
  fixed seeds with shaping off. The first version of this project, a plain 300k-step PPO, scored
  51.1.
- **How it improved:** six multi-seed experiments with random training seeds
  ([the write-ups](docs/evidence/experiments/README.md)). A richer observation and an
  LSTM were the clear wins; layout randomisation did not help.
- **The honesty:** it still collides about 4x more than random, the shipped model is the best of 5
  seeds (a typical run scores about 60), it generalises poorly to maze-like layouts, and the
  [limitations](#what-this-does-and-does-not-show) are written down.

## Why I Built This

I wanted to learn multi-agent RL properly, and that meant not starting from a ready-made
environment. A learned policy is only as trustworthy as the world it trains in and the baseline it
is compared against. So I wrote the rules, rewards, observations and renderer myself against the
PettingZoo `ParallelEnv` API, tested them, added two scripted baselines, trained an agent, and then
ran controlled experiments to find out which changes really helped and which only looked like they did.

## Results

Every policy is scored on the same 100 seeded episodes (seeds 1000 to 1099), 3 robots, 200 steps,
shaping off, with 95% confidence intervals.

| Policy | Deliveries / episode | Collisions / episode |
|--------|---------------------:|---------------------:|
| random |        2.63 ± 0.34   |        12.80 ± 1.49  |
| greedy |       12.67 ± 2.07   |       401.30 ± 24.24 |
| **Tuned recurrent PPO (shared)** | **61.99 ± 1.37** | **50.90 ± 4.91** |
| *First version: plain PPO, 300k steps (`ppo_v1_300k.zip`)* | *51.07 ± 3.27* | *189.95 ± 20.65* |

- The tuned PPO delivers about **4.9x more than greedy and 24x more than random**, and collides about
  one eighth as often as greedy.
- Greedy has the worst collision count by far. A collision is counted per robot per step, so 200
  steps with 3 robots allows at most 600, and greedy sits at about 400: its rule only looks at its
  own target and four neighbours, so robots spend much of the episode blocking each other.
- Random barely collides because it barely goes anywhere. Collisions on their own are not a quality
  measure, which is why every result reports both columns.

### How the model improved

![Deliveries per episode while training: tuned versus the plain PPO baseline](docs/evidence/learning_curve_deliveries.png)

The tuned model starts slower but overtakes the plain PPO by about 100k to 200k steps. At the same
1M-step budget it is ahead by 4.6 deliveries (about 2.8 standard errors), with about 3.7x fewer
collisions. The plain PPO keeps improving to about 61 at 3M steps, so this is an equal-budget
comparison, not a claim to beat a fully trained plain PPO.

| Experiment | Question | Finding |
|---|---|---|
| [Rollout budget](docs/evidence/experiments/00-rollout-budget.md) | Was SB3's default rollout size hurting? | `n_steps = 128` gave 55 to 60 deliveries against 41 |
| [E0: seed spread](docs/evidence/experiments/e0-seed-spread.md) | Is the committed 51 typical? | Yes: 5 seeds give 51.1 ± 4.6, range 44 to 57 |
| [E1: longer training](docs/evidence/experiments/e1-longer-training.md) | Where does plain PPO plateau? | About 61 at 2M to 3M steps; arg-max actions collapse to about 13 |
| [E2: hyperparameter sweep](docs/evidence/experiments/e2-hyperparameter-sweep.md) | Which knobs matter? | Collision penalty -0.3 and entropy 0.01 help (borderline); -1.0 hurts; shaping, discount, width: no clear effect |
| [E3: observation and memory](docs/evidence/experiments/e3-observation-and-memory.md) | Do richer observations and an LSTM help? | Yes, clearly: +7 (observation), +10 (LSTM), +12 (both) at 500k steps |
| [E4: generalisation](docs/evidence/experiments/e4-generalisation.md) | Does training on random layouts transfer? | No: 8 deliveries worse on held-out layouts |
| [E5: tuned model](docs/evidence/experiments/e5-tuned-model.md) | Does the combination beat the baseline and ship? | 60.0 vs 55.4 at 1M steps; passed the replacement rule |

**Sampled versus deterministic.** The plain PPO collapses when it takes the arg-max action (about 11
deliveries, the same as greedy). The tuned recurrent policy mostly does not (55 deliveries), though
sampling is still better (60). The cause is not diagnosed; memory plausibly lets the policy notice it
has been blocked. See [E1](docs/evidence/experiments/e1-longer-training.md) and
[E3](docs/evidence/experiments/e3-observation-and-memory.md).

![Sampled versus deterministic deliveries during training, tuned model](docs/evidence/sampled_vs_deterministic_tuned.png)

**Generalisation.** Both variants were trained on the default layout with 3 robots only, then scored
on every layout and 2 to 6 robots. On the two layouts neither had seen, the tuned model averaged 42.7
deliveries against 34.7 for the plain PPO. On the default layout its total deliveries rise with more
robots (44 at 2 robots, 83 at 6), though per-robot throughput falls (about 22 to 14). The maze-like
layout (`maze_lite`) stays poor for both (22 and 13).

![Deliveries by layout and robot count](docs/evidence/generalisation.png)

### What this does and does not show

- **Shown:** on this layout, with 3 robots, a single shared network trained with PPO beats both
  scripted baselines on deliveries and beats greedy on collisions, with confidence intervals that do
  not overlap. A richer observation and a recurrent policy each clearly helped, in 5-seed
  experiments where every non-baseline seed beat every baseline seed.
- **Not shown:** strong generalisation. The shipped model was trained on one map with 3 robots. It
  transfers well to a similar wider-gap map but stays poor on a maze-like one, and training on random
  layouts did not fix that ([E4](docs/evidence/experiments/e4-generalisation.md)).
- **Several changes at once.** The tuned model bundles four changes. The experiments support the
  memory and observation changes clearly and the collision-penalty and entropy changes only
  tentatively; the final run did not isolate them.
- **Selection and budget.** The shipped model is the best of 5 seeds, picked on a validation block
  and checked on a separate test block (63.0); the other four seeds average about 60.5. It costs about
  30x more training time per step than the plain network, and it was compared at 1M steps, where a
  plain PPO at 3M reaches about 61.
- **Open problem:** it still collides about 51 times per episode, about 4x random. I have not
  investigated where those collisions happen, so I make no claim about why.
- **Shaping caveat:** potential-based shaping preserves the optimal policy exactly only with no
  discounting and within one target's lifetime. PPO discounts at 0.99 and targets change on delivery,
  so the guarantee holds only approximately here.
- **Seed policy:** training seeds are random and recorded in each experiment's `manifest.json`, but
  `runs/` is git-ignored, so the seeds and evaluation blocks used are written in
  [`FINDINGS.md`](docs/evidence/results/FINDINGS.md) instead.

The full list of caveats is in [`docs/limitations.md`](docs/limitations.md). To rerun anything, see
[`docs/reproducing.md`](docs/reproducing.md); for how the code fits together, see
[`docs/architecture.md`](docs/architecture.md).

## The RL Approach

### Problem formulation

| Piece | Choice | Why it matters for learning |
|---|---|---|
| Setting | Simultaneous-move, partially observable, with individual rewards | Robots act at once and see only their neighbourhood, so coordination has to emerge from local information |
| Observation | `Box(-1, 1, (8,))` by default: own row and column, offset to target, four blocked flags (wall, shelf or another robot). The shipped model uses the optional 14-value `rich` mode, which adds four robot-neighbour flags and the offset to the nearest other robot | Small enough to learn from quickly, yet it contains what is needed to avoid a neighbour |
| Actions | `Discrete(5)`: stay, up, down, left, right | Discrete, so PPO's categorical policy applies directly |
| Reward | +1.0 delivery, -0.01 per step, -0.1 per collision (environment default; the shipped model was trained with -0.3) | The step penalty rewards speed and the collision penalty is the only signal about other robots |
| Episode | 200 steps, all robots truncated together | A fixed horizon makes episodes comparable |

### Training pipeline

1. **PPO from Stable-Baselines3.** I chose not to re-implement the optimiser: the point is to
   understand and apply it, and to demonstrate the environment.
2. **Parameter sharing with SuperSuit.** The robots are identical, so one network controls all of
   them. SuperSuit turns each robot into one sub-environment of a vectorised env, and 8 copies are
   concatenated: `3 robots x 8 envs = 24` sub-environments feeding one network, so every robot's
   experience improves every robot's behaviour.
3. **Reward shaping, in training only.** The +1 delivery reward is sparse, so a randomly wandering
   robot rarely sees it. I add a potential-based bonus of `+0.02 x (distance to target before the
   step - after)` (Ng et al., 1999). It is on during training and **always off in evaluation**.
4. **Truncation bootstrapping.** SuperSuit merges "terminated" and "truncated" into one `done` flag, so
   SB3 would treat the step-200 cut-off as the end of the future. The environment now sets
   `TimeLimit.truncated` on the last step so SB3 bootstraps the value there. (The `legacy` variant
   keeps the old behaviour for comparison; at 300k steps the difference was within noise.)
5. **Memory.** The shipped model uses sb3-contrib `RecurrentPPO` with an LSTM policy, so a robot can
   remember its recent past and escape standoffs.
6. **Stochastic actions at inference.** The trained policy samples from its action distribution
   instead of taking the arg-max. Seeding keeps each sampled episode reproducible.

```bash
pip install -e ".[rl]"     # add ",plot" for `experiment plot`
python -m multi_agent_warehouse.train   # plain PPO, 300k steps, random seed; writes models/ppo_retrained.zip
python -m multi_agent_warehouse.experiment run --name my-run --variants tuned --seeds 1 --steps 1000000
```

`train` never writes over the shipped model (its default output is `models/ppo_retrained`). Its
defaults are a plain PPO with truncation bootstrapping and a random seed, which is not the
first-version recipe. `train --seed 0 --no-bootstrap-truncation` uses the first version's settings; I
have not checked that it reproduces the committed `ppo_v1_300k.zip` bit for bit.

The shipped model, `models/ppo_multi_agent_warehouse.zip` (7.2 MB), is the `tuned` variant (training
seed 330, 1M steps). Its sidecar, `models/ppo_multi_agent_warehouse.json`, holds its training config,
which `evaluate.py` and `play.py` read to build the matching observation
and policy class. A model without a sidecar is treated as a plain PPO on the basic observation. The
RL dependencies are an optional extra, imported lazily, so the environment and scripted baselines
install without PyTorch.

### Experiment method

Training seeds are drawn at random from `[0, 1000)` and recorded in each experiment's
`manifest.json`; evaluation blocks of 100 consecutive seeds are drawn from `[10,000, 2^31)`, so they
never overlap a training seed, and every variant in an experiment is scored on the same block. A
difference counts as real only if it is more than twice its standard error. `select` picks the best
seed on one random validation block and reports it on a second, disjoint test block, so the reported
score is not inflated by the choice. The full method, the variant table and the commands are in
[`docs/evidence/experiments/`](docs/evidence/experiments/README.md).

## Challenges & What I Learned

**A budget problem hidden by a default**
- *Problem:* SB3's default rollout size looked fine, but `n_steps` counts per sub-environment and
  parameter sharing multiplies those.
- *Approach:* Work out the arithmetic before training: 24 sub-environments make the default rollout
  49,152 samples, so 300,000 steps give only about 7 updates. Then test smaller rollouts across
  several seeds rather than trusting one run.
- *Result:* `n_steps = 128` raised deliveries from 41 to about 55 to 60 ([write-up](docs/evidence/experiments/00-rollout-budget.md)).

**One run proves little**
- *Problem:* The first README number came from one training seed.
- *Approach:* Re-run the same recipe with 5 random seeds and measure the spread, then require every
  later claim to be a multi-seed comparison on a shared evaluation block.
- *Result:* 51 turned out typical but ±5, which means most single-run "improvements" below that size
  would have been noise ([E0](docs/evidence/experiments/e0-seed-spread.md)).

**A hypothesis that failed**
- *Problem:* I expected training on many layouts to help on unseen ones.
- *Approach:* Hold out two layouts, score both variants on all of them, and pre-commit to reporting
  the gap either way.
- *Result:* It was 8 deliveries worse on held-out layouts, mostly because the "held-out" `wide_gap`
  is nearly the default map. Reporting that honestly, and leaving the idea out of the final model,
  was the right call ([E4](docs/evidence/experiments/e4-generalisation.md)).

**Selecting a model without cheating**
- *Problem:* Picking the best of 5 seeds on the same episodes you report on inflates the score.
- *Approach:* Choose on a validation block, report on a disjoint test block, and set the rule for
  replacing the shipped model before looking at the result.
- *Result:* The selected seed scored 63.0 on the test block while the others averaged 60.5, which is
  why the README says a typical run is about 60 ([E5](docs/evidence/experiments/e5-tuned-model.md)).

**Seeding through a vectorised wrapper**
- *Problem:* `PPO(..., seed=...)` crashes because SuperSuit's `ConcatVecEnv` has no `seed` method.
- *Approach:* Seed Python, NumPy and PyTorch through `set_random_seed`, and seed the environments
  through `ConcatVecEnv.reset(seed=...)`.
- *Result:* The same seed gives identical weights and a different seed gives different weights,
  which is what makes the multi-seed comparisons meaningful.

**Simultaneous moves are harder than they look**
- *Problem:* Two robots can target one cell, swap, or be blocked indirectly by a robot that was
  itself blocked.
- *Approach:* Start with the two obvious rules, notice that a blocked robot becomes a stayer and can
  block others, and iterate to a fixed point.
- *Result:* A small pure function with its own tests, and a clear definition of when a robot
  "collided", which the collision penalty and metric both depend on.

**Takeaways:** Most of RL lives in details that never appear in an algorithm diagram: what the
observation hides, which events a reward can see, how defaults interact with vectorisation, and how
seeding and evaluation blocks decide whether two runs are comparable at all.

## The Environment

| Part | Choice |
|---|---|
| Interface | PettingZoo `ParallelEnv`; all robots act at once |
| Map | 8 x 8; the default layout has four 2 x 2 shelf blocks. Five more layouts (`open`, `aisles`, `scatter`, `wide_gap`, `maze_lite`) exist for generalisation tests |
| Robots | 1 to 6 (default 3), named `robot_0`, `robot_1`, ... |
| Task | Reach your target; on delivery you immediately get a new one |
| Episode | No terminations; every robot is truncated together at 200 steps |

- **Complete API:** `reset`/`step`/`render`/`close`, per-agent spaces, truncation, `info` dicts,
  strict input validation, and seeded reproducibility from a single `np.random.default_rng`.
- **Physics as a pure function:** `resolve_moves` settles simultaneous moves with two rules (a cell
  two robots want, and a swap). A blocked robot counts as a stayer, so the rules repeat until
  nothing changes. Bumping a shelf or the edge just keeps the robot in place and is not a collision.
- **Testable rendering:** the renderer always draws offscreen and returns an RGB array; the window is
  an optional extra layer, so rendering is tested headless.
- **Options:** `observation="rich"`, `collision_penalty`, `shaping_scale` and `layouts=(...)`. The
  defaults, and the random stream with a single layout, are unchanged from the first version (pinned
  by tests).

A step runs in a fixed order: validate actions, resolve moves, check targets, compute rewards,
advance the clock, then build observations. Observations are built last so a robot that just
delivered already sees its new target.

## Architecture

```
config.py   layout(s) + every tunable number (the only place settings live)
rules.py    resolve_moves(): pure function, no env state
env.py      MultiAgentWarehouseEnv: ParallelEnv built on rules + config
render.py   Renderer: always draws offscreen, returns an RGB array; "human" also blits to a window
agents.py   Policy protocol (reset, act) + RandomPolicy, GreedyPolicy, RLPolicy, model sidecars
train.py    trains the shared PPO / recurrent PPO policy (optional `rl` extra)
seeds.py    random training and evaluation seeds
experiment.py  multi-seed experiments, generalisation runs, model selection on a test block
plots.py    learning curves, comparison figures and reports (optional `plot` extra)
plots_generalisation.py  layout-by-robot-count heatmaps
evaluate.py seeded evaluation table with confidence intervals
play.py     live demo with keyboard controls
```

### Key design decisions

| Decision | Alternatives considered | Why I chose this |
|---|---|---|
| `ParallelEnv` | AEC (turn-based) | Robots act simultaneously, and Parallel maps directly onto vectorised training |
| Small local observation (8 numbers, 14 in `rich` mode) | Full-grid or windowed image | Small enough to learn from quickly; `rich` was added only after an experiment showed it helps |
| One shared network (parameter sharing) | Independent networks per robot; centralised-critic methods (MAPPO, QMIX) | Identical robots, so one network learns from all experience; the simplest multi-agent training that works with single-agent libraries |
| Stable-Baselines3 PPO (and sb3-contrib `RecurrentPPO`) | Hand-written PPO | The goal is to apply and evaluate PPO, not to write an optimiser |
| Shaping off by default, never in evaluation | Always on | Keeps reported numbers about the task, not the training aid |
| Policies see observations only | Policies read env internals | Keeps scripted and learned policies interchangeable behind one interface |
| Collisions via a pure function | Resolve inside `step` | A pure function is testable on its own and can be reasoned about without env state |
| Settings in one `config.py` | Constants per module | Changing the map or rewards is a one-file edit |
| Random seeds, shared evaluation blocks, 2·SE rule | One fixed seed; eyeballing curves | A fixed seed lets a lucky run pass as a result; shared blocks pair the comparison |

## Getting Started

Requires Python 3.10+.

```bash
git clone https://github.com/Wayne-cyw/multi-agent-warehouse.git
cd multi-agent-warehouse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,rl,plot]"  # drop ",rl" to skip PyTorch (scripted baselines only); "plot" is for experiment figures
```

```bash
python play.py --policy rl                  # watch the trained PPO agent
python play.py                              # greedy baseline (space pause, R reset, P switch policy, Q quit)
python play.py --policy random --robots 4   # other policy, more robots
python -m multi_agent_warehouse.evaluate    # reproduce the results table (--episodes N changes the sample size)
python -m multi_agent_warehouse.train       # train a plain PPO (writes models/ppo_retrained.zip and its sidecar)
pytest                                      # run the tests
```

`play.py` falls back to greedy if the model file is missing. The `rl` row in the evaluation table
appears only when the model exists and its libraries (Stable-Baselines3, plus sb3-contrib for the
shipped recurrent model) are installed.

## Project Structure

```
multi-agent-warehouse/
├── multi_agent_warehouse/
│   ├── config.py     # layouts and settings
│   ├── rules.py      # resolve_moves
│   ├── env.py        # the ParallelEnv
│   ├── render.py     # Pygame renderer
│   ├── agents.py     # random, greedy and RL policies
│   ├── train.py      # PPO / recurrent PPO training
│   ├── seeds.py      # random seeds
│   ├── experiment.py # multi-seed experiments, model selection
│   ├── plots.py      # learning curves and comparison figures
│   ├── plots_generalisation.py
│   └── evaluate.py   # seeded evaluation
├── models/           # shipped model, the first-version model, and config sidecars
├── play.py           # live demo
├── tests/            # one test file per module
└── docs/
    ├── README.md                           # index of the docs
    ├── architecture.md                     # module map and data flow
    ├── reproducing.md                      # commands, seeds and expected numbers
    ├── limitations.md                      # everything the results do not show
    ├── design-decisions.md                 # every decision and its trade-offs
    ├── code-style.md
    ├── evidence/experiments/               # one write-up per experiment
    ├── evidence/results/                   # FINDINGS.md and the experiment figures
    └── study-notes/                        # learning notes
```

## Testing & Quality

- 200 tests covering the rules, the environment contract (invalid actions, calls after the episode
  ends, bad robot counts, determinism), the renderer, the demo, the PPO and recurrent-PPO stack
  (training saves a loadable model and sidecar, the policy acts for every robot), the seed and
  experiment pipeline (disjoint evaluation blocks, resumability, model selection) and the plots. The
  RL tests skip if the `rl` extra is absent, and everything runs headless via
  `SDL_VIDEODRIVER=dummy`.
- `ruff` (rules `E, F, I, B, UP`, line length 100) and `ruff format --check`.
- Seeded evaluation: identical calls return identical numbers, and tests pin the original
  environment's random stream so the first-version results stay reproducible.

## Roadmap

- [x] Environment, rules and rewards
- [x] Pygame renderer and live demo
- [x] Random and greedy baselines with a seeded evaluation and confidence intervals
- [x] Train PPO (Stable-Baselines3) with parameter sharing through SuperSuit, using shaping during training only
- [x] Compare the learned policy against both baselines, and report honestly where it lands
- [x] Improve training with multi-seed experiments (longer training, memory, richer observation, reward and entropy tuning)
- [x] Test generalisation across layouts and robot counts (measured; maze-like layouts are still poor)
- [ ] Analyse where PPO's remaining collisions happen
- [ ] Isolate the collision-penalty and entropy contributions to the tuned model, and train longer than 1M steps
- [ ] Generalise better: observations that do not reveal the absolute position, more held-out layouts
