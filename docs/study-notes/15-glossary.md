# 15. Glossary

[Index](README.md) · Previous: [14 generalization](14-generalization.md) · Next: [16 exercises](16-exercises.md)

Terms are grouped by topic. The number in brackets is the note that explains the idea.

## Multi-agent ideas ([02](02-multi-agent-systems.md))

| Term | Definition |
| --- | --- |
| **Agent** | An entity that observes, acts and is rewarded. Here: a robot. |
| **Cooperative / competitive / mixed** | Agents share goals / have opposed goals / some of each. |
| **CTDE** | Centralised training, decentralised execution: use extra global information to train, act from local observations. |
| **Credit assignment** | Working out which agent's action deserves the credit or blame for a team outcome. |
| **Dec-POMDP** | Multi-agent problem with partial observations and no central controller at run time. |
| **Individual reward** | Each agent receives its own reward (this project). |
| **Non-stationarity** | The environment appears to change because other agents are learning. |
| **Observation / state** | What one agent sees / the full truth of the world. |
| **Parameter sharing** | All agents use one set of network weights ([07](07-parameter-sharing-and-supersuit.md)). |
| **Partial observability** | An agent sees only part of the situation. |

## RL and PPO ([03](03-reinforcement-learning-basics.md), [04](04-ppo.md))

| Term | Definition |
| --- | --- |
| **Action value Q(o, a)** | Expected return after taking action `a` first. |
| **Actor / critic** | The policy network / the value network in actor-critic methods such as PPO. |
| **Advantage A** | `Q - V`: how much better an action was than the policy's average from that situation. |
| **Bootstrapping** | Estimating the future return with a learned value instead of waiting for it. |
| **Clipping** | PPO's limit on how far the probability ratio may move the objective, `[1 - epsilon, 1 + epsilon]`. |
| **Discount (gamma)** | Weight of future reward against immediate reward. |
| **Entropy** | How spread out the action probabilities are; high means more exploration. |
| **Episode** | One run from `reset()` to termination or truncation. |
| **Exploration / exploitation** | Trying new actions / using what already works. |
| **GAE** | Generalised Advantage Estimation: a blend of TD errors controlled by `lambda`. |
| **Logits** | The actor's raw outputs before softmax. |
| **Minibatch / epoch** | A slice of the rollout buffer used for one gradient step / one pass over the buffer. |
| **On-policy** | Learns only from data collected by the current policy, then discards it. |
| **Policy pi** | Mapping from observation to action (or to action probabilities). |
| **PPO** | Proximal Policy Optimization: a stable actor-critic, policy-gradient algorithm. |
| **Probability ratio r** | `pi_new(a given o) / pi_old(a given o)` for the action that was taken. |
| **Return** | Cumulative (discounted) reward. |
| **Rollout buffer** | The data collected between PPO updates: `n_steps x number of sub-environments` samples. |
| **Softmax** | Turns raw numbers into probabilities that sum to 1. |
| **Sparse reward** | Reward that arrives rarely (here: only on delivery). |
| **TD error** | `r + gamma * V(next) - V(now)`: how surprised the critic was. |
| **Terminated / truncated** | Episode ended by the task itself / cut off by a limit. |
| **Value function V(o)** | Expected return from an observation under the current policy. |

## Reward shaping ([05](05-reward-shaping.md))

| Term | Definition |
| --- | --- |
| **Manhattan distance** | `|dr| + |dc|`: grid steps with no diagonals. |
| **Potential-based shaping** | Extra reward `gamma * Phi(s') - Phi(s)`; does not change the optimal policy. |
| **Reward hacking** | The agent maximises the reward signal in a way the designer did not intend. |
| **Shaping** | Extra reward that guides learning. |
| **Telescoping** | Terms that cancel in pairs, leaving only the ends. |

## PettingZoo and Gymnasium ([06](06-pettingzoo-and-gymnasium.md))

| Term | Definition |
| --- | --- |
| **AEC** | PettingZoo's turn-based API (one agent acts per step). |
| **Box / Discrete** | Gymnasium spaces: an array in a range / an integer 0 to n-1. |
| **Gymnasium** | The standard single-agent RL environment API (the maintained successor to OpenAI Gym). |
| **Parallel API** | PettingZoo API where all agents act simultaneously. |
| **`possible_agents` / `agents`** | Every agent that can exist / the ones alive right now. |
| **SuperSuit** | Wrappers that adapt PettingZoo environments for libraries such as SB3. |
| **Vectorised env** | Many environment copies stepped together to collect data faster. |

## This project ([08](08-warehouse-rules.md), [09](09-move-resolution.md), [10](10-rendering-pygame.md))

| Term | Definition |
| --- | --- |
| **Blit** | Copy one image onto another (surface to window). |
| **Cascade** | One blocked robot becoming a stayer that blocks the next. |
| **Collision** | A robot that tried to move and was blocked by another robot. A wall bump is not a collision. |
| **Delivery** | Ending a step on one's target. |
| **Egocentric observation** | Observation expressed relative to the agent itself. |
| **Fixed point** | A state where repeating a process changes nothing (how move resolution stops). |
| **HUD** | The status bar drawn above the grid. |
| **Layout** | The `LAYOUT` map in `config.py`. |
| **Offscreen-first** | Always draw to memory; optionally copy to a window. |
| **Pure function** | Output depends only on inputs; no side effects (`resolve_moves`). |
| **Stayer** | A robot that does not move this step (chose to stay, bumped a wall, or was blocked). |
| **Truncation** | The episode ends at `max_steps`, for all robots at once. |

## Generalization ([14](14-generalization.md))

| Term | Definition |
| --- | --- |
| **Distance field** | For every cell, its path length to a target (from a breadth-first search). |
| **Held-out layout** | A map never used in training, kept for evaluation. |
| **In-distribution / out-of-distribution** | Like the training data / unlike it. |
| **Overfitting** | Doing well on the training maps but not on new ones. |
| **Reactive policy** | Chooses from the current observation only, with no memory. |
