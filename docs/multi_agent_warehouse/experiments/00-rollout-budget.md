# Rollout budget (the first tuning, before the experiment pipeline)

## Question

Stable-Baselines3 collects `n_steps` steps from every sub-environment before each policy update.
SuperSuit turns every robot into its own sub-environment, so with 8 copies of a 3-robot warehouse
there are `3 x 8 = 24` sub-environments. Was the default `n_steps = 2048` too large for a 300,000
step budget?

## Setup

- Plain PPO, 300,000 steps, the three settings below, 30 evaluation episodes from seed 1000,
  shaping off. This was run by hand, before the multi-seed pipeline existed, so it uses training
  seeds 0 to 2 (not random draws) and only 30 episodes.

## Result

With 2048, one rollout is `2048 x 24 = 49,152` samples, so 300k steps give only about 7 policy
updates. Smaller rollouts give about 98 updates from the same data.

| `n_steps` / `batch_size` | Training seeds | Deliveries / episode | Collisions / episode |
|---|---|---|---|
| 2048 / 64 (SB3 defaults) | 0 | 41.4 | 35.7 |
| 256 / 256 | 0, 1, 2 | 50.8, 57.2, 49.8 | 189.6, 126.1, 189.6 |
| **128 / 256 (chosen)** | 0, 1, 2 | 54.7, 58.5, 60.1 | 170.9, 136.3, 136.9 |

## Takeaways

- `n_steps = 128`, `batch_size = 256` delivered the most and varied the least across seeds, and is
  the default in every later experiment.
- The default setting delivers less but collides far less (36 against 137 to 171). That is a
  different point on the speed-versus-safety curve, and was not explored at the time.

## Caveats

- Three seeds and 30 episodes: enough to see a 40 versus 55 gap, not enough to rank 54.7 against
  58.5. The later experiments redo this kind of comparison properly.
