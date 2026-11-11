# Experiments

Every claim about the trained agent comes from one of the experiments below. Each has its own page
with the question it asked, the exact command, the numbers, the figures and the caveats. The raw
numbers and the long-form analysis are in [`../results/FINDINGS.md`](../results/FINDINGS.md).

| Page | Question | Headline |
|---|---|---|
| [Rollout budget](00-rollout-budget.md) | Was the SB3 default rollout size hurting the first model? | `n_steps = 128` beat the default: about 55 to 60 deliveries against 41 |
| [E0: seed spread](e0-seed-spread.md) | Is the single committed result (51.07) typical or lucky? | Typical: 5 seeds give 51.1 ± 4.6 |
| [E1: longer training](e1-longer-training.md) | Where does the plain PPO stop improving? | About 61 deliveries at 3M steps; sampled actions needed |
| [E2: hyperparameter sweep](e2-hyperparameter-sweep.md) | Which training knobs matter? | Collision penalty -0.3 and entropy 0.01 help (borderline); -1.0 hurts |
| [E3: observation and memory](e3-observation-and-memory.md) | Do a richer observation and an LSTM help? | Yes, clearly: +7 and +10 deliveries, +12 combined |
| [E4: generalisation](e4-generalisation.md) | Does training on random layouts transfer to unseen ones? | No: it did worse on the held-out layouts |
| [E5: tuned model](e5-tuned-model.md) | Does the combined recipe beat the baseline, and should it ship? | 60 against 55 at 1M steps, and it replaced the committed model |

## How every experiment is run

- **Random training seeds.** Seeds are drawn at random from 0 to 999 and written to the
  experiment's `manifest.json`, so a lucky run cannot pass as a result and any run can still be
  repeated (`--train-seeds`).
- **One shared exam.** Each experiment draws one random block of 100 consecutive evaluation seeds
  (always at or above 10,000, so it never overlaps a training seed). Every variant, seed and
  checkpoint is scored on that block, so differences between variants are not episode luck.
- **Raw metrics only.** Evaluation always runs with reward shaping off and reports deliveries and
  collisions per episode. The training reward is never used to judge progress.
- **Sampled actions are the main metric.** Deterministic (arg-max) evaluation is recorded as well
  (see [E1](e1-longer-training.md) for why it is not the headline).
- **The verdict rule.** For a variant against `base`: Δ = variant mean minus `base` mean over seeds,
  SE = sqrt(s_v²/n + s_b²/n). Δ > 2·SE is "helps", Δ < -2·SE is "hurts", anything else is "no clear
  effect". With 5 seeds a single crossing of 2·SE happens by chance about 8 to 10% of the time, so
  borderline results are marked tentative.
- **Checkpoints.** Snapshots are taken at the end of whole rollouts, so "50k" is really 52,224 steps.
  The tables use the real step counts.
- **Resumable.** A finished run leaves `final.zip`; re-running the same command skips finished runs.

## Commands

```bash
pip install -e ".[dev,rl,plot]"
python -m multi_agent_warehouse.experiment run --name NAME --variants V1,V2 --seeds 5 --steps 1000000
python -m multi_agent_warehouse.experiment plot --name NAME
python -m multi_agent_warehouse.experiment generalise --name NAME --variants V1,V2
python -m multi_agent_warehouse.experiment select --name NAME --variant V1
```

Run outputs go to `runs/experiments/NAME/` (git-ignored): `manifest.json`, `eval_curve.csv`,
`baselines.csv`, per-run training logs and snapshots, `plots/` and `report.md`. The numbers and
figures that matter are copied into `docs/multi_agent_warehouse/results/`.

## Variants

All variants equal `base` in every field they do not mention. `base` is a plain PPO with an MLP
policy (64, 64), 8 parallel environments, `n_steps` 128, `batch_size` 256, learning rate 3e-4,
discount 0.99, entropy coefficient 0, collision penalty -0.1, shaping scale 0.02, 3 robots,
truncation bootstrapped.

| Variant | Change from `base` | Used in |
|---|---|---|
| `legacy` | no truncation bootstrapping (the original behaviour); 300k steps | E0 |
| `base` | (the reference) | E1 to E5 |
| `ent_0.01` | entropy coefficient 0.01 | E2 |
| `shape_0`, `shape_0.05`, `shape_0.1` | shaping scale 0, 0.05, 0.1 | E2 |
| `gamma_0.95`, `gamma_0.995` | discount 0.95, 0.995 | E2 |
| `net_128` | network (128, 128) | E2 |
| `coll_0.3`, `coll_1.0` | collision penalty -0.3, -1.0 | E2 |
| `rich_obs` | 14-value observation | E3 |
| `lstm` | `RecurrentPPO` with an LSTM policy | E3 |
| `rich_lstm` | both of the above | E3 |
| `fixed` | the same as `base`, named for E4's comparison | E4 |
| `rand` | random layouts and 2 to 5 robots during training | E4 |
| `tuned` | `rich_lstm` + collision penalty -0.3 + entropy 0.01 | E5 |
