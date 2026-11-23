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



3. Consider a batch with two responses: one short (3 response tokens and 3 masked tokens) and one long (6 response tokens and 0 masked tokens). Assume similar per-token log-prob ratio and advantages, and set the fixed maximum response length to 6 for the masked_mean_drgrpo reduction.

3.1. Apply each reduction to the batch and call backward(), then compare the gradient w.r.t. the per-token log-prob ratio. What do you observe?

Under masked_mean, each token of the short response gets a gradient of 1/6 and each token of the long response 1/12, so a short response's tokens weigh twice as much.
Under masked_mean_drgrpo, every response token gets a gradient of 1/12, whatever the length of its response.
The masked tokens get no gradient under either.

3.2. Implications: Give an example scenario where masked_mean is preferable to masked_mean_drgrpo, and one where the opposite is true. Give a brief explanation for each.

Where `masked_mean_drgrpo` is better - LLMs do next token prediction, and training an LLM is getting the LLM to predict the better next token given the current capability of the model. If we are biased by length, we are teaching the model that the token leading to the shorter response is better than the token leading to the longer response, even though the truth might not be the case. The bias also runs the other way for wrong responses: a long wrong response is penalized less per token than a short one, so we are also teaching the model that a wrong response is less bad if it is longer.

Where `masked_mean` is better - For example we are sampling the model with a low temperature, and longer sequences perform better than shorter sequences on average. With `masked_mean`, each response carries a total weight proportional to its advantage, and the advantages in a group sum to zero, so the push up on the better responses balances the push down on the worse ones. With `masked_mean_drgrpo`, a response's total weight also scales with its length, so if the longer responses are the better ones, the update has a net push up on the model's own samples. This is like training the model on its own samples, which makes the model predict the more likely token even more and results in mode collapse, faster at a low temperature.


