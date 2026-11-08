# Findings: seed spread and learning curves (E0, E1)

## Settings actually used

- Machine: 10 cores, 16 GB; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; code at commit `be71c10`.
- Throughput (benchmark, default env, no `WAREHOUSE_CONTRACTS`): 300k steps in about 14 s, so roughly 21k env steps/s
  (60k steps: 4.4 s including about 2.5 s of import time). `cProfile` of 20k steps: `inspect.getmodule` is 0.13 s of 2.9 s (about 4.5%), so the contract wrapper does not dominate.
  One evaluation of 100 episodes for all three policies (random, greedy, rl) takes 3.7 s in total.
- E0 `legacy` (300k steps, truncation not bootstrapped), 5 seeds. Training seeds 886, 523, 717, 522, 538; eval seed base 536374503 (100 episodes). Wall clock 2 min 47 s (training and evaluation).
- E1 `base` (truncation bootstrapped), 5 seeds, full 3M steps (the 3 hour budget rule did not trigger). Training seeds 199, 731, 848, 179, 31; eval seed base 672125400 (100 episodes). Wall clock 13 min 8 s.
- Seeds were drawn at random by the runner and are in each `manifest.json`. E0 and E1 use different eval blocks, so E0-vs-E1 comparisons are unpaired.
- Checkpoint step counts are rounded up to whole rollouts (3072 samples each), so the nominal 50k, 100k, 200k, 300k, 500k, 1M, 2M, 3M checkpoints are really 52,224, 101,376, 202,752, 301,056, 500,736, 1,001,472, 2,002,944, 3,001,344. Tables use the nominal names.
- All numbers are the mean over the 5 seeds of each seed's 100-episode mean; "±" is the standard deviation across the 5 seeds (n-1).

## (a) Spread of `legacy` at 300k (E0, sampled)

Deliveries 51.12 ± 4.55 (min 44.44, max 57.26); collisions 172.16 ± 32.21. The committed result (51.07 deliveries, 189.95 collisions, one run) sits at the mean of the 5-seed distribution for deliveries and inside the collision spread (about 0.5 std above the mean). So 51.07 is typical, not a lucky draw, but a single run can differ by about ±5 deliveries (range 13 across seeds) and ±30 collisions.

## (b) Does truncation bootstrapping move the 300k score?

E1 `base` at 300k: 47.81 ± 4.41 deliveries, 207.14 ± 29.47 collisions; E0 `legacy` at 300k: 51.12 ± 4.55, 172.16 ± 32.21. The difference (-3.3 deliveries) is smaller than either seed std and the standard error of the difference is about 2.9, so it is within seed noise (and unpaired: different eval blocks). There is no evidence that bootstrapping helps at 300k; if anything the point estimate is lower. Both variants show the same early dip (see below), so bootstrapping alone does not remove it.

## (c) Where do deliveries stop improving (E1, sampled)?

The curve is not monotone: 53.5 at 50k, 53.7 at 100k (essentially the early plateau, reached within about 50k steps), then a decline to 50.5 at 200k and a trough of 47.8 at 300k, a slow recovery to 50.3 at 500k and 53.9 at 1M, then a clear rise to 59.6 at 2M and 60.8 at 3M. From 2M to 3M the gain (+1.2) is smaller than the seed std (1.4 to 1.8), so the curve looks saturated by about 2M, but only from two points; I did not train beyond 3M. Seed spread is smallest at 2M and 3M (std 1.4 and 1.8, versus 4.4 to 5.5 at 300k and 1M). The 1M checkpoint spans 46.9 to 61.7 across seeds, i.e. the seeds escape the plateau at different times. Final E1 (3M): 60.78 ± 1.77, versus E0's 51.12 ± 4.55 at 300k; the larger gain comes from training much longer, not from bootstrapping (see b). The E0 curve (300k) is flat or declining after 100k (54.7 at 100k, 51.1 at 300k).

## (d) Sampled versus deterministic

Deterministic is far worse at every checkpoint and every seed, and collisions are higher.

| checkpoint | sampled deliveries | deterministic deliveries | sampled collisions | deterministic collisions |
|---|---|---|---|---|
| 50k | 53.54 ± 1.48 | 8.71 ± 0.58 | 61.40 ± 7.33 | 255.45 ± 16.81 |
| 100k | 53.71 ± 2.06 | 10.83 ± 1.08 | 136.99 ± 10.60 | 302.41 ± 25.63 |
| 200k | 50.50 ± 2.11 | 11.77 ± 0.86 | 184.42 ± 14.55 | 322.41 ± 7.56 |
| 300k | 47.81 ± 4.41 | 12.14 ± 0.70 | 207.14 ± 29.47 | 343.46 ± 15.42 |
| 500k | 50.26 ± 2.81 | 12.63 ± 0.84 | 194.89 ± 23.73 | 355.58 ± 19.33 |
| 1M | 53.94 ± 5.50 | 12.38 ± 0.71 | 190.89 ± 32.54 | 355.00 ± 13.24 |
| 2M | 59.62 ± 1.42 | 12.98 ± 0.38 | 160.13 ± 14.42 | 383.69 ± 6.78 |
| 3M | 60.78 ± 1.77 | 13.00 ± 0.86 | 157.20 ± 10.15 | 393.02 ± 13.92 |

E0 `legacy` rows (50k / 100k / 150k / 200k / 250k / 300k), sampled deliveries: 50.44 ± 1.92, 54.70 ± 1.03, 52.66 ± 2.88, 52.95 ± 1.80, 52.34 ± 2.98, 51.12 ± 4.55; sampled collisions: 57.35, 107.31, 149.88, 157.77, 164.45, 172.16; deterministic deliveries: 7.57, 9.18, 10.37, 11.12, 10.89, 11.08; deterministic collisions: 226.85, 277.19, 323.40, 337.26, 329.42, 330.98 (the 300k row is from the E0 report).

The deterministic policy delivers roughly the same as the greedy baseline (13.5 to 14.1 deliveries, 391 to 411 collisions) and approaches its collision count (393 at 3M). This policy is evidently stochastic by necessity: the argmax action presumably gets robots stuck against each other, so evaluating the deterministic policy would understate the trained agent. The gap does not close with training (it widens from 45 to 48 deliveries), and deterministic collisions rise with training (255 to 393). I did not investigate the mechanism beyond this; the behaviour is observed, not diagnosed.

## (e) Do collisions fall as deliveries rise?

Not early. Over the first 300k steps (both experiments), sampled collisions rise steadily from about 60 to 170 to 207 while deliveries stay flat or fall; the agent is moving more, not more carefully. Only in the later phase do both move the right way: from 300k to 3M in E1, deliveries rise from 47.8 to 60.8 while collisions fall from 207 to 157 (190.9 at 1M, 160.1 at 2M). The scatter of final checkpoints shows all 5 seeds in the high-deliveries, 143 to 168 collisions corner, far from greedy (about 410 collisions). Collisions after 3M (157) are still about 12 times the random baseline (13.5), so collisions are reduced from their peak but not low.

## Concerns and caveats

- E0 vs E1 comparisons are unpaired (different eval blocks); only differences clearly larger than the seed std should be read as real.
- Only 5 seeds per experiment; stds are rough.
- Saturation at about 2M to 3M is read from two last points; longer training was not run.
- The initial plateau and dip (about 50k to 300k) appears in both variants and in the training reward diagnostic (peak about 15 at 80k, drop to about 9 to 10, slow recovery over 1M+ steps).

Figures: `e1/learning_curve_deliveries.png`, `e1/learning_curve_collisions.png`, `e1/sampled_vs_deterministic_base.png`, `e1/diagnostics_base.png`.
