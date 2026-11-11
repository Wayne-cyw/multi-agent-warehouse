# Limitations

What this project does not show, and what is still untested. The README's
["What this does and does not show"](../README.md#what-this-does-and-does-not-show) is the short
version; this page collects the caveats from every experiment in one place. Each item names where
the evidence is.

## What the results support

On the default layout with 3 robots, one shared network trained with PPO beats both scripted
baselines on deliveries, and beats greedy on collisions, with non-overlapping 95% confidence
intervals. A richer observation and a recurrent policy each clearly helped in 5-seed experiments
([E3](evidence/experiments/e3-observation-and-memory.md)).

## Scope of the task

- **One map, three robots.** The shipped model was trained on the default layout only. It transfers
  to a similar wider-gap layout but stays poor on the maze-like one, and training on random layouts
  did not fix that ([E4](evidence/experiments/e4-generalisation.md)).
- **Only two held-out layouts**, and one of them (`wide_gap`) is nearly the default map, so the
  held-out result is weaker evidence than it looks.
- **The observation contains the absolute cell**, so a policy can memorise the default map. This is
  a plausible reason generalisation is poor; it was not tested.
- **A small, synthetic task.** 8 x 8 grid, fixed 200-step episodes, no pick-up or drop-off, no
  battery, no dynamics. Results say nothing about real warehouse robotics.

## Statistical strength

- **Five seeds per variant.** Standard deviations are rough, and only differences of about 5 to 6
  deliveries are detectable in the sweep ([E2](evidence/experiments/e2-hyperparameter-sweep.md)).
- **Two borderline "helps" results.** Collision penalty -0.3 and entropy 0.01 crossed the 2·SE line,
  but nine comparisons were made, so about 0.7 to 0.9 chance crossings were expected. Only the
  "hurts" result (collision penalty -1.0) is credible.
- **Unpaired cross-experiment comparisons.** Each experiment draws its own evaluation block, so
  E0 to E5 numbers are comparable only loosely. Within one experiment all variants share a block.
- **The first rollout-budget experiment** used three seeds and 30 episodes: enough to see 40 against
  55, not to rank 54.7 against 58.5 ([write-up](evidence/experiments/00-rollout-budget.md)).

## The shipped model

- **Four changes at once.** `tuned` bundles the rich observation, the LSTM, a -0.3 collision penalty
  and entropy 0.01. E3 supports the first two clearly; the last two were only borderline in E2, and
  E5 did not isolate any of them ([E5](evidence/experiments/e5-tuned-model.md)).
- **Selection bias.** It is the best of 5 seeds. It scored 63.0 on a separate test block, but the
  other four seeds average about 60.5, so a typical run is about 60, not 62.
- **Equal budget, not equal quality.** `tuned` at 1M steps is roughly level with the plain PPO at 3M
  (about 61). What is shown is that `tuned` is better and far safer at equal and small budgets.
- **Cost.** About 30x more training time per step than the plain network, and a 7.2 MB model
  against 146 KB. At equal wall-clock time the plain PPO could train much longer.
- **Memory and size are confounded.** The LSTM also has many more parameters. A bigger MLP did not
  help in E2, which weakly suggests capacity alone is not the reason; it was not tested directly.
- **The 300k first version** was trained for less, so comparing it with the shipped model mixes
  method with training budget.

## Open problems

- **Collisions.** The shipped model still collides about 51 times per episode, about 4x random.
  Where and why they happen has not been investigated, so no claim is made about the cause.
- **Sampled versus deterministic actions.** The plain PPO collapses under arg-max (about 11
  deliveries, level with greedy); the recurrent policy mostly does not (55). The cause is not
  diagnosed. A plausible, untested explanation is that robots that always take their most likely
  action mirror each other into a head-on standoff, and that memory lets a policy notice it has been
  blocked ([E1](evidence/experiments/e1-longer-training.md)).
- **Learning-curve dips.** The plain PPO plateaus early and dips (about 54 at 100k, 48 at 300k)
  before recovering, and the tuned model dips slightly at the end (62.7 at 500k, 60.0 at 1M).
  Neither was investigated.
- **Checkpoint step counts.** Snapshots land at the end of whole rollouts, so "50k" is really
  52,224 steps; the tables use the real counts.

## Method caveats

- **Shaping guarantee is approximate.** Potential-based shaping preserves the optimal policy exactly
  only with no discounting and within one target's lifetime. PPO discounts at 0.99 and targets
  change on delivery, so the guarantee holds only approximately here.
- **Reproducibility.** Training seeds are random by design and `runs/` is git-ignored, so the seeds
  and evaluation blocks are recorded in the docs instead
  ([`reproducing.md`](reproducing.md)). Exact reproduction of the committed zip files has not been
  checked, and results depend on library versions.
- **Collisions are not a quality measure on their own.** A robot that barely moves barely collides
  (random), so every result reports deliveries and collisions together.

## Not yet done

From the README roadmap: analyse where the remaining collisions happen; isolate the collision-penalty
and entropy contributions and train beyond 1M steps; generalise better (observations that hide the
absolute position, more held-out layouts).
