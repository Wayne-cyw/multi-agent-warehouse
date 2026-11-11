# Documentation index

Start with the [root README](../README.md) for the pitch, the results table and the limitations.
This folder holds the detail behind it.

| Path | What it is | Read it when |
|---|---|---|
| [`architecture.md`](architecture.md) | Module map, the order of a step, and how data flows from config to evaluation | You want to read or change the code |
| [`reproducing.md`](reproducing.md) | Commands, seeds and expected numbers to regenerate the results | You want to check a claim yourself |
| [`limitations.md`](limitations.md) | Every caveat and untested hypothesis, gathered from all the experiments | You want to know how far to trust a result |
| [`design-decisions.md`](design-decisions.md) | Every design decision with its reasons, rejected alternatives and trade-offs | You ask "why was it done this way?" |
| [`code-style.md`](code-style.md) | Docstring, runtime-checking and comment conventions | You are contributing code |
| [`evidence/experiments/`](evidence/experiments/README.md) | One write-up per experiment (rollout budget, E0 to E5) | You want the question, command and caveats behind a number |
| [`evidence/results/FINDINGS.md`](evidence/results/FINDINGS.md) | The long-form numbers and analysis, plus the seeds and evaluation blocks used | You want the raw figures |
| [`study-notes/`](study-notes/README.md) | Seventeen short notes on the ideas behind the project, with a glossary and exercises | A term or technique is unfamiliar |

## Reading order

1. The root README, for what was built and what it does and does not show.
2. `study-notes/` if multi-agent RL or PPO is new to you.
3. `architecture.md`, then the code.
4. `experiments/` and `FINDINGS.md` for the evidence, `limitations.md` for how far to trust it, `design-decisions.md` for the reasoning.

## Images

The README embeds the headline figures that sit directly in `evidence/`
(`demo.gif`, `generalisation.png`, `learning_curve_*.png`, `sampled_vs_deterministic_tuned.png`).
The same plots, per experiment, are in `evidence/results/e1` to `e5`. The top-level
copies are deliberate: they are the ones the README links to.

Raw experiment output (`runs/`) is git-ignored; the numbers and figures that matter are copied into
`evidence/results/`.
