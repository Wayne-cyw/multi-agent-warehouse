# 17. Further reading

[Index](README.md) · Previous: [16 exercises](16-exercises.md)

> Titles, authors and arXiv numbers below are **from memory**. Confirm them when you open the
> papers.

## Books and courses

- **Sutton and Barto**, *Reinforcement Learning: An Introduction* (2nd ed.): free online. Chapters
  1 to 3 cover [03](03-reinforcement-learning-basics.md); chapter 13 covers policy gradients (the
  background for [04](04-ppo.md)).
- **OpenAI Spinning Up** (spinningup.openai.com): short, clear explanations of policy gradients and
  PPO.

## PPO and policy gradients ([04](04-ppo.md))

- **PPO**: Schulman et al., 2017, *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- **GAE**: Schulman et al., 2015, *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation*, arXiv:1506.02438.
- **TRPO**: Schulman et al., 2015, *Trust Region Policy Optimization*, arXiv:1502.05477 (the idea PPO
  simplifies).
- **The 37 Implementation Details of PPO** (ICLR Blog Track, 2022): why details such as advantage
  normalisation matter.
- **Stable-Baselines3 docs** (stable-baselines3.readthedocs.io): the PPO page lists the parameters and
  defaults used in the notes. Check them against the installed version.

## Multi-agent RL ([02](02-multi-agent-systems.md), [07](07-parameter-sharing-and-supersuit.md))

- **PettingZoo**: Terry et al., 2021, arXiv:2009.14471; docs at pettingzoo.farama.org.
- **MADDPG** (CTDE): Lowe et al., 2017, arXiv:1706.02275.
- **MAPPO**: Yu et al., 2021, *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*,
  arXiv:2103.01955: PPO with parameter sharing works well in cooperative tasks.
- **Gymnasium docs**: gymnasium.farama.org.

## Reward shaping ([05](05-reward-shaping.md))

- **Ng, Harada and Russell**, 1999, *Policy Invariance Under Reward Transformations: Theory and
  Application to Reward Shaping*, ICML.

## Multi-robot path finding and generalization ([14](14-generalization.md))

- **PRIMAL**: Sartoretti et al., 2019, *PRIMAL: Pathfinding via Reinforcement and Imitation
  Multi-Agent Learning* (local view plus a goal direction, trained with a shared policy). Search for
  it by title; I am not certain of the arXiv number.
