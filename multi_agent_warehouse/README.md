# Multi-Agent Warehouse

A small PettingZoo + Pygame warehouse where several robots pick up and deliver items on an 8 x 8
grid. Two scripted policies (random and greedy) share one interface, so the live demo and the
evaluation run the same loop. The larger project is `warehouse_rl/`.

![Greedy robots delivering items](../docs/multi_agent_warehouse/demo.gif)

## Install and run

```bash
pip install -e .
python play.py
python -m multi_agent_warehouse.evaluate
pytest
```

## Evaluation

Mean per episode over 20 seeds (1000 to 1019), 3 robots, shaping off:

| policy | deliveries/episode | collisions/episode |
|--------|-------------------:|-------------------:|
| random |               2.45 |               9.90 |
| greedy |              13.50 |             366.90 |

## Design

- `MultiAgentWarehouseEnv` is a PettingZoo `ParallelEnv`: all robots act at once each step.
- Moves are resolved by a pure two-rule function (`resolve_moves` in `rules.py`).
- Each robot sees an 8-value observation: position, vector to its target, four blocked flags.
- The renderer always draws offscreen and returns an RGB array; `human` mode also blits to a window.
- Policies use only observations and never read env internals.
