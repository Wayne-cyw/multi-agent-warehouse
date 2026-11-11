# E0: seed spread of the original recipe

## Question

The README originally reported 51.07 deliveries per episode from one training run (seed 0). Is that
a typical result for the recipe, or a lucky or unlucky draw?

## Setup

- Variant `legacy`: the original recipe, 300,000 steps, truncation not bootstrapped.
- 5 random training seeds (886, 523, 717, 522, 538), recorded in `manifest.json`.
- One shared evaluation block (base seed 536374503, 100 episodes), sampled actions.
- Wall clock: 2 min 47 s.

```bash
python -m multi_agent_warehouse.experiment run --name e0-legacy --variants legacy --seeds 5 --steps 300000
```

## Result

| | Deliveries | Collisions |
|---|---|---|
| 5-seed mean ± seed std | 51.12 ± 4.55 | 172.16 ± 32.21 |
| min to max | 44.44 to 57.26 | |
| committed single run | 51.07 | 189.95 |

The committed number sits almost exactly at the mean, so it was typical, not lucky. A single run of
this recipe can still land anywhere in a range of about 13 deliveries.

## What it means

- A one-run comparison cannot detect an improvement smaller than about ±5 deliveries. Every later
  experiment therefore uses 5 seeds and reports the spread.
- E0 sets the yardstick: the "old model" is 51 ± 5.

## Caveats

- Five seeds give a rough standard deviation.
- E0 used a different evaluation block from E1, so E0-against-E1 comparisons are unpaired.

Full detail: [`FINDINGS.md`](../results/FINDINGS.md) section (a).
