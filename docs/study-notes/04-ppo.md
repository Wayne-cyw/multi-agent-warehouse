# 04. PPO (Proximal Policy Optimization)

[Index](README.md) · Previous: [03 RL basics](03-reinforcement-learning-basics.md) · Next: [05 reward shaping](05-reward-shaping.md)

PPO is the learning algorithm for the project's stretch stage (Stage 5). You will **not**
implement it: the project uses Stable-Baselines3's `PPO("MlpPolicy", ...)`. These notes explain what
that one line does, so you can read training logs, choose settings and diagnose problems.

> **Status of facts here.** Stable-Baselines3 2.9.2 is now installed (Task 7). The algorithm
> explanation is standard. The PPO defaults quoted below were checked against `PPO.__init__`'s
> signature on 2026-11-07 (`n_steps` 2048, `batch_size` 64, `n_epochs` 10, `gamma` 0.99,
> `gae_lambda` 0.95, `clip_range` 0.2, `ent_coef` 0.0, `vf_coef` 0.5, `learning_rate` 3e-4); the network
> shape and log-line details marked "from memory" were not. Worked numbers marked *(computed)* were
> calculated, not run through SB3.

## 1. Where PPO sits

```
REINFORCE          basic policy gradient: noisy, learns from whole-episode returns
   |
Actor-critic       add a learned value function V(o) to reduce noise
   |
TRPO               limit how far each update can move the policy (complicated maths)
   |
PPO                get most of that stability with a simple clipped objective
```

PPO is a **policy-gradient**, **actor-critic**, **on-policy** method:

- **Policy-gradient**: it improves the policy directly instead of learning a value for each action.
- **Actor-critic**: it learns two things, a policy (the *actor*) and a value function (the *critic*).
- **On-policy**: it learns only from data collected by the *current* policy and throws that data
  away after each update.

## 2. The policy gradient idea

Say a robot, when its observation has "target is to the right and right is not blocked", chose
action RIGHT and soon delivered. We want the policy to be *more likely* to choose RIGHT in similar
situations. Say it chose LEFT and collided; we want LEFT *less* likely.

The policy-gradient rule says exactly that:

```
increase the log-probability of an action in proportion to how good the action turned out to be

gradient  ~  E[ grad log pi(a | o)  x  A ]
```

`A` is the **advantage**: how much better the action was than what the policy normally gets from
that situation. Positive advantage: make it more likely. Negative: less likely. Zero: leave it.

## 3. The two networks

For this project the observation is 8 numbers and there are 5 actions.

```
                     +--> actor  --> 5 numbers ("logits") --> softmax --> probabilities --> sample an action
 8 observation  -----+
 numbers             +--> critic --> 1 number: V(o), the expected return from here
```

- The actor's output is turned into probabilities with **softmax**. During training the action is
  *sampled* from that distribution (this is the exploration).
- The critic predicts the return. It is used only for training; at run time the actor does all the work.
- `"MlpPolicy"` means both are **multi-layer perceptrons** (fully connected networks). SB3's default
  is, from memory, two hidden layers of 64 units each, with `tanh`, kept **separate** for actor and
  critic. Small networks are appropriate for an 8-number input.

## 4. Advantage and GAE

How do we know "how much better than usual" an action was? We use the critic.

**One-step TD error** (how surprised the critic was):

```
delta_t = r_t + gamma * V(o_{t+1}) - V(o_t)
```

If the robot got a bigger reward or landed somewhere more valuable than the critic expected,
`delta` is positive.

**Generalised Advantage Estimation (GAE)** blends TD errors from the next several steps:

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
```

`lambda` (0 to 1) trades **bias for variance**:

| `lambda` | Behaviour |
| --- | --- |
| 0 | `A_t = delta_t`. Uses one step plus the critic's guess. Low noise, but biased if the critic is wrong. |
| 1 | `A_t = (actual return) - V(o_t)`. No dependence on the critic's guess about the future, but noisy. |
| 0.95 (SB3 default, from memory) | Mostly the long view, with some smoothing. |

**Worked example** *(computed, with `gamma = 1` for clarity)*. Three steps, rewards `0, 0, 1`, critic
values `V = 0.2, 0.4, 0.7`, and `V = 0` after the end:

```
delta_0 = 0 + 0.4 - 0.2 = 0.2
delta_1 = 0 + 0.7 - 0.4 = 0.3
delta_2 = 1 + 0   - 0.7 = 0.3

lambda = 1:   A = [0.8, 0.6, 0.3]     (= return - V:  1-0.2, 1-0.4, 1-0.7)
lambda = 0:   A = [0.2, 0.3, 0.3]     (= the TD errors)
```

Same data, different views. SB3 also **normalises advantages** within each minibatch by default
(from memory), so only their relative size matters.

## 5. The clipped objective: the "proximal" in PPO

A naive policy-gradient step on old data could change the policy so much that the data no longer
describes it, and training collapses. PPO prevents this by comparing the new policy to the one that
collected the data.

**Probability ratio** for an action that was actually taken:

```
r = pi_new(a | o) / pi_old(a | o)         r = 1 means "unchanged"; r = 1.5 means "50% more likely now"
```

**Clipped surrogate objective** (we *maximise* it), with `epsilon = 0.2` (SB3 default, from memory):

```
L_clip = mean over samples of  min( r * A ,  clip(r, 1 - epsilon, 1 + epsilon) * A )
```

`clip(r, 0.8, 1.2)` forces `r` into `[0.8, 1.2]`. The `min` makes the objective **pessimistic**: once
the policy has moved far enough in the helpful direction, there is no more reward for moving further.

**Worked examples** *(computed)*, `epsilon = 0.2`:

| Advantage A | Ratio r | `r * A` | clipped `* A` | `min` | What happens |
| --- | --- | --- | --- | --- | --- |
| +2 (good action) | 1.5 | 3.0 | 1.2 x 2 = 2.4 | **2.4** | Already 50% more likely. The clipped term is a constant, so **gradient is zero**: stop pushing. |
| +2 (good action) | 0.5 | 1.0 | 0.8 x 2 = 1.6 | **1.0** | Policy made a good action *less* likely. Unclipped term is chosen: **gradient active**, push back up. |
| -2 (bad action) | 1.5 | -3.0 | 1.2 x -2 = -2.4 | **-3.0** | Policy made a bad action *more* likely. **Gradient active**, push it down. |
| -2 (bad action) | 0.5 | -1.0 | 0.8 x -2 = -1.6 | **-1.6** | Bad action already 50% less likely. Clipped term chosen, **gradient zero**: stop pushing. |

Reading the table: PPO lets the policy move toward better behaviour by up to about 20% per action
probability per update, then stops. It never refuses to *correct* a mistake.

## 6. The full loss

SB3 minimises one combined loss:

```
loss = - L_clip  +  vf_coef * (V(o) - return)^2  -  ent_coef * entropy
```

| Term | Purpose | SB3 default (from memory) |
| --- | --- | --- |
| `-L_clip` | Improve the policy, safely | |
| value loss | Train the critic to predict return (mean squared error) | `vf_coef = 0.5` |
| entropy bonus | Reward uncertainty, to keep exploring | `ent_coef = 0.0` (off) |

**Entropy** measures how spread out the action probabilities are (max for 5 equal actions,
`ln 5 = 1.61`; zero if one action has probability 1). With `ent_coef = 0` nothing stops the policy
becoming deterministic early. In this project that is a real risk: deterministic identical robots
can lock into a standoff ([02](02-multi-agent-systems.md)). If you see that, a small `ent_coef`
(for example 0.01) is the first thing to try.

## 7. The training loop

```
repeat until total_timesteps is reached:
    1. COLLECT   run the current policy in every sub-environment for n_steps steps each
                 -> a "rollout buffer" of (obs, action, reward, ...) of size n_steps x n_envs
    2. ADVANTAGES   use the critic to compute GAE advantages and returns for the buffer
    3. UPDATE    for n_epochs passes over the buffer:
                    shuffle, split into minibatches of batch_size
                    for each minibatch: compute the loss above and take one gradient step
    4. DISCARD   the buffer (the data is now "old policy" data)
```

**Illustrative sketch** (pseudocode, not project code and not SB3's code):

```python
for iteration in range(num_iterations):
    buffer = collect_rollouts(policy, vec_env, n_steps)           # on-policy data
    advantages, returns = gae(buffer, critic, gamma, lam)
    old_log_probs = buffer.log_probs                               # frozen snapshot of pi_old
    for epoch in range(n_epochs):
        for batch in minibatches(buffer, batch_size):
            log_probs, values, entropy = policy.evaluate(batch.obs, batch.actions)
            ratio = (log_probs - batch.old_log_probs).exp()
            adv = normalise(batch.advantages)
            policy_loss = -torch.min(ratio * adv,
                                     ratio.clamp(1 - eps, 1 + eps) * adv).mean()
            value_loss = ((values - batch.returns) ** 2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
            optimizer.zero_grad(); loss.backward()
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
```

## 8. SB3's defaults and what they mean for this project

*(The numeric defaults below were confirmed against the installed SB3 2.9.2 signature on 2026-11-07.)*

| Setting | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | 3e-4 | Size of each gradient step |
| `n_steps` | 2048 | Steps collected **per sub-environment** before each update |
| `batch_size` | 64 | Minibatch size for the gradient steps |
| `n_epochs` | 10 | Passes over each rollout buffer |
| `gamma` | 0.99 | Discount |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.2 | `epsilon` above |
| `ent_coef` | 0.0 | Entropy bonus weight |
| `vf_coef` | 0.5 | Value-loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |

### A budget problem the plan does not yet mention *(computed)*

The plan trains `PPO("MlpPolicy", vec_env, seed=seed, verbose=0)` with `n_envs = 8` and
`total_steps = 300_000`, and leaves `n_steps` at its default. Parameter sharing through SuperSuit
makes **each robot its own sub-environment** ([07](07-parameter-sharing-and-supersuit.md)):

```
sub-environments      = 3 robots x 8 copies                    = 24
samples per rollout   = n_steps x sub-environments = 2048 x 24 = 49,152
policy updates        = ceil(300,000 / 49,152)                 = 7
real episodes seen    = 300,000 / 3 robots / 200 steps         = 500
```

So 300,000 steps would be only about **7 policy updates** from about **500 episodes**: very little
for PPO. The plan allows retraining with 1,000,000 steps, but the cheaper fix is to lower `n_steps`
(for example 128 to 256 gives many more updates from the same data), or raise `total_steps`.
This was settled in Task 7 (decision D-52 in [`../design-decisions.md`](../design-decisions.md)).
The arithmetic above is a calculation. The **measurement** (300,000 steps, 30 evaluation episodes,
shaping off) was: random 2.4 deliveries per episode, greedy 13.6, and PPO 41.4 with the defaults
(`n_steps = 2048`), 49.8 to 57.2 with `n_steps = 256`, and 54.7 to 60.1 with `n_steps = 128`
(`batch_size = 256`, seeds 0 to 2). Every setting cleared the 2x-random bar, so the small budget was
not a blocker, but `n_steps = 128` was highest and steadiest and is now the default in `train.py`.

## 9. Why PPO fits (and strains) this project

| Property | Fit |
| --- | --- |
| Discrete actions (5) | Natural: softmax over 5 logits |
| Small, normalised observation (8 values in `[-1, 1]`) | Good: no input normalisation needed |
| Small rewards (-0.1 to +1) | Fine, since advantages are normalised |
| Parameter sharing | All robots' transitions go in one buffer and are treated as independent samples |
| On-policy, stationary environment assumed | Mildly violated: other robots' behaviour changes as the shared policy changes ([02](02-multi-agent-systems.md)). In practice PPO with shared weights works well in cooperative settings (the MAPPO paper, [17](17-further-reading.md)) |
| Sample efficiency | PPO is not sample efficient (data is discarded). Fine here because the simulator is cheap |

## 10. Reading the training log

Run with `verbose=1` while learning (the plan uses `verbose=0`). Key lines from SB3's output, from memory:

| Metric | Healthy sign | Warning sign |
| --- | --- | --- |
| `rollout/ep_rew_mean` | Rising over time | Flat at the random-policy level |
| `train/explained_variance` | Rising toward 1 (the critic predicts returns) | Near 0 or negative for long |
| `train/approx_kl` | Small (about 0.01 or less) | Large spikes: updates too big |
| `train/clip_fraction` | Around 0.05 to 0.2 | Near 0 (barely learning) or very high (too aggressive) |
| `train/entropy_loss` | Gradually shrinking in magnitude | Collapsing to 0 quickly: policy went deterministic too early |

Note that `ep_rew_mean` is measured *with shaping on* (the plan trains with shaping on), so compare
policies with `evaluate.py` (shaping off), never with the training reward ([12](12-evaluation.md)).

## 11. Common failure modes

| Symptom | Likely cause | First things to try |
| --- | --- | --- |
| Robots learn to always `STAY` | The per-step penalty `-0.01` is tiny and collisions hurt more than waiting; delivery was never discovered | Keep shaping on; more training; check that random already delivers sometimes |
| Robots stand off in a head-on meeting | Deterministic identical policies | Sample actions; add a small `ent_coef` |
| Reward rises in training, not in evaluation | The agent exploits shaping (hovers near the target) | Lower `SHAPING_SCALE`; check delivery rate directly |
| Training reward noisy and not improving | Too few updates (section 8) | Smaller `n_steps`; more steps |
| Works in the demo, fails on another map | Overfit to one layout | [14 generalization](14-generalization.md) |

## Check yourself

1. What does the advantage tell the policy, and where does the critic come in?
2. In the clip table, why is the gradient zero when `A = +2` and `r = 1.5`, but active when `A = +2`
   and `r = 0.5`?
3. What does `lambda = 0` versus `lambda = 1` change in GAE?
4. Why is PPO "on-policy", and what does it do with the data after each update?
5. With 3 robots, 8 copies and `n_steps = 256`, how many samples are in one rollout, and how many
   updates does 300,000 steps give? (Answer: 6,144 and about 49.)
6. Why might `ent_coef = 0` be a problem specifically for identical robots?
7. Why must you evaluate with shaping off, even though you trained with it on?
