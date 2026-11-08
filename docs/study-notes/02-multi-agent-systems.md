# 02. Multi-agent systems

[Index](README.md) · Previous: [01 overview](01-project-overview.md) · Next: [03 RL basics](03-reinforcement-learning-basics.md)

## What is an agent?

An **agent** observes part of the world, chooses an action, and receives a reward signal. A
**multi-agent system** has more than one agent acting in the *same* world at the *same* time. In
this project each robot is an agent: `robot_0`, `robot_1`, `robot_2`.

## Why multi-agent is harder than single-agent

| Difficulty | Plain-English meaning | Where it shows up here |
| --- | --- | --- |
| **Non-stationarity** | From one robot's view the world keeps changing because the *other robots are also learning*. What worked yesterday may fail today. | RL stage: three robots share one network that is being updated |
| **Partial observability** | Each robot sees only part of the situation. | A robot sees its own position, its own target and four neighbour-blocked flags. It does **not** see other robots' targets or anything beyond adjacent cells |
| **Credit assignment** | If the team succeeds or fails, who deserves the credit or blame? | Rewards are *individual* here, which sidesteps the problem (see below) |
| **Coordination** | Good behaviour requires agents to avoid conflicts or cooperate. | Two robots heading into one cell both get penalised |
| **Scalability** | The joint action space grows exponentially: 5 actions with 3 robots is 5^3 = 125 joint actions. | Why we use one per-robot policy instead of one policy over joint actions |

## Agent relationships

- **Cooperative**: agents share a goal and usually a reward (a team).
- **Competitive**: one agent's gain is another's loss (zero-sum games like chess).
- **Mixed**: some of both.

**This project is "independent and loosely cooperative".** Each robot is rewarded for its *own*
deliveries, but they share a floor, so they get in each other's way. They never need to help each
other, only to *not interfere*. That makes it a gentle first multi-agent problem.

## Individual vs shared reward

| | Individual reward (this project) | Shared / team reward |
| --- | --- | --- |
| Signal | Each robot gets its own delivery bonus and collision penalty | One number for the whole team |
| Pros | Clear learning signal, easy credit assignment | Aligns everyone with the team goal |
| Cons | Robots may be selfish (for example, never yield) | Noisy: hard to tell whose action mattered |

## Observation vs state

- **State**: the full truth of the world (all positions, all targets, all shelves).
- **Observation**: what one agent actually gets to see.

When agents see only their own observation and must act without a central controller, the setting is
a **Dec-POMDP** (decentralised, partially observable Markov decision process). Ours is a simple case:
each robot acts from only its own 8-number observation (see [08](08-warehouse-rules.md)).

## Three ways to train multi-agent RL

| Approach | Idea | Used here? |
| --- | --- | --- |
| **Independent learners** | Each agent has its own network and treats the others as part of the environment | No |
| **Parameter sharing** | **One** network for all agents. Each agent feeds in its own observation and gets its own action; every agent's experience trains the same weights | **Yes (Stage 5)** |
| **Centralised training, decentralised execution (CTDE)** | Train with extra global information (for example a central critic); at run time each agent acts on its own observation | No (named so you know the term: MADDPG, MAPPO, QMIX) |

Details of parameter sharing, and how it is implemented, are in
[07](07-parameter-sharing-and-supersuit.md).

**Why parameter sharing fits here**: the robots are *identical* (same observation format, same
actions, same goal), so one network can learn a single "how to be a warehouse robot" skill, and it
sees every robot's experience.

**Its limitation**: identical robots given identical observations choose identical actions. In a
head-on standoff two robots might both step forward forever. The project handles this by *sampling*
actions rather than always taking the most likely one (see [11](11-policies-and-demo.md)).

## Check yourself

1. Why is the collision penalty a *multi-agent* feature rather than just a navigation one?
2. A robot sees an adjacent robot only as a blocked flag. Which multi-agent difficulty is that?
3. What is the trade-off between individual and team reward?
4. Why can three identical robots with one shared policy get into a standoff?
5. Why does the joint action space make "one policy over all robots" unattractive?
