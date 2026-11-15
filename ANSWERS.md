1. In the PPO-style loss above, is clipping more critical for on-policy or off-policy updates in this setup? Briefly justify your answer in terms of the probability ratio.

Clipping is more critical for off-policy updates.
On-policy, the rollouts come from the current policy and we take one gradient step per batch, so π_θold = π_θ and ratio_t = 1 for every token. That is always inside [1−ε, 1+ε], so clipping never activates and the update is just the plain policy gradient (REINFORCE with the group baseline).
Off-policy, the rollouts come from an older policy, so ratio_t drifts away from 1. Once a token's ratio moves past 1+ε (A_t > 0) or below 1−ε (A_t < 0), clipping zeroes its gradient. This keeps the new policy close to the one that generated the data.

2. In the context of GRPO for this task, list two advantages of on-policy and two advantages of off-policy and give a brief explanation for each.

Advantages of on-policy updates.

- You are training the model on what it could have done better, not on what another (older) model could have done better. The samples come from π_θ itself, so A·∇log π_θ is an unbiased estimate of the current policy's gradient. Off-policy needs an importance ratio to correct for the mismatch, and clipping that ratio adds bias.
- On-policy data reflects how the current model performs, not how an old model performed. As the model learns Countdown, its mistakes change. At L256, early on, most failures are responses cut off before <answer>. Later, the model finishes in time but gets the arithmetic wrong. On-policy rollouts always target the failure mode the model has now.

Advantages of off-policy updates.
- Better hardware utilization. The inference engine keeps generating with slightly stale weights while training runs, instead of waiting for each backward pass and weight reload.
- This allows rollouts to be reused for several gradient steps. This especially helps early at L256, where successful rollouts are rare and worth extracting more signal from.
