# Training Improvement & Experiment Pipeline: Design

## Goal

Find out how much the committed PPO agent (51.07 ± 3.27 deliveries/episode, one run, seed 0) can be
improved, and produce data and graphs that show how the model improves over the course of training.
Every claim must come from several randomly seeded runs, not one.

## Findings that shape the design

- **One run proves little.** The README number comes from a single training seed. The spread between
  seeds is unknown, so it is measured first (variant `legacy`).
- **Truncation is treated as a real episode end.** SuperSuit's SB3 wrapper merges `terminated` and
  `truncated` into `done` and never sets `TimeLimit.truncated`; SB3 only bootstraps the value at the
  time limit when that key is present (`on_policy_algorithm.py`, around line 240). The observation has
  no clock, so the agent is told the future is worth 0 at step 200 even though nothing changed. The
  fix is one info key; `legacy` keeps the old behaviour for comparison.
- **Training reports no reward curve today.** There is no `Monitor`, so SB3 logs no
  `rollout/ep_rew_mean`. `VecMonitor` plus SB3's CSV logger give the training-side curves.

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Training seeds | Drawn at random from 0-999 (OS entropy) when not given; always written to `manifest.json`; `--seeds` or `--seed` reproduces a run | "Seeds should be randomized" without losing reproducibility |
| Evaluation seeds | One random block of 100 consecutive seeds per experiment, shared by every variant, seed, checkpoint and baseline in it; the training seeds avoid that block | Paired comparison: differences between variants are not episode noise |
| README table | `python -m multi_agent_warehouse.evaluate` keeps seeds 1000-1099 | The committed table stays reproducible |
| Model selection | Choose the best seed on a validation block, report on a separate fresh test block | Picking the best of N seeds on the reported seeds would inflate the score |
| Step 5 reward change | Only the training reward changes; evaluation still reports raw deliveries and collisions | The metrics stay comparable across variants |
| Observation change (step 6) | `rich` = the existing 8 values + 4 robot-only blocked flags + offset to the nearest other robot (14 values) | The first 8 keep their meaning, so `GreedyPolicy` is unchanged; the old model is rejected with a clear error |
| Recurrent policy (step 7) | `sb3-contrib` `RecurrentPPO`, `MlpLstmPolicy` | Memory to break deadlocks |
| Randomisation (step 8) | Mixed layouts (fixed 8x8) and mixed robot counts, scored on held-out layouts and counts | Directly answers the roadmap's generalisation item |

## Experiments

| Id | Variants | Seeds | Steps | Shows |
|---|---|---|---|---|
| E0 | `legacy` (committed setup, no truncation bootstrap) | 5 | 300k | Seed spread of the current result |
| E1 | `base` (truncation fixed) | 5 | 3M | Learning curve, saturation, sampled vs deterministic |
| E2 | `base` plus one-factor changes: `ent_0.01`, `shape_0`, `shape_0.05`, `shape_0.1`, `gamma_0.95`, `gamma_0.995`, `net_128`, `coll_0.3`, `coll_1.0` | 3 | 1M | Which knobs matter |
| E3 | `base`, `rich_obs`, `lstm`, `rich_lstm` | 3 | 1M | Observation and memory |
| E4 | `rand` vs the same config trained on the default map only | 3 | 1M | Generalisation to held-out layouts and counts |
| E5 | `tuned` (winners from E2-E4) | 5 | 3M | Final model |

Budgets are a ceiling: a throughput benchmark runs first and the steps are reduced if E1 would take
more than about 3 hours.

## Outputs per experiment (`runs/experiments/<name>/`)

- `manifest.json`: seeds, eval seed blocks, variant configs, git commit, library versions.
- `eval_curve.csv`: variant, seed, timesteps, mode (`sampled`/`deterministic`), mean and 95% CI of
  deliveries and collisions.
- `baselines.csv`: random and greedy on the same eval seeds.
- `<variant>/seed_<s>/progress.csv`: SB3 training log (episode reward, entropy, value loss,
  explained variance).
- `plots/`: learning curves (deliveries, collisions) with per-seed lines, a mean and 95% band, and
  baseline reference lines; training diagnostics; sampled vs deterministic; variant comparison;
  deliveries-vs-collisions trade-off; generalisation heatmap.
- `report.md`: the tables and figure links. Final figures are copied to
  `docs/multi_agent_warehouse/` for the README.

## Non-goals

- Larger or differently sized grids, MAPPO / centralised critics, GPU training.
- Changing the committed model unless the final test block shows a clear gain over it.
