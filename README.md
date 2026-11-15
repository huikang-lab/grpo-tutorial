# GRPO for Countdown

In this assignment you will implement **GRPO** and **Dr.GRPO** and train `Qwen/Qwen3-1.7B` to solve Countdown problems: reach a target number using the given numbers and `+ - * /`.

The handout is `gstar_assignment3.pdf`.

## Setup

```bash
bash setup_env.sh              # creates .venv: Python 3.11, torch 2.7.1 (CUDA 12.8), transformers 4.51.3, datasets
source .venv/bin/activate
```

## Run

```bash
# 1. Implement Tasks 1-2, then check them against the handout's asserts and get the zero-shot baselines:
python zero_shot_eval.py --max_tokens 256        # also 512 and 1024  (expect ~0.2%, ~15%, ~38%)

# 2. Implement Tasks 3-5, then train (80 steps, eval every 10):
python starter.py --loss_type grpo    --max_tokens 256
python starter.py --loss_type dr_grpo --max_tokens 256
python starter.py --loss_type grpo    --max_tokens 512
python starter.py --loss_type dr_grpo --max_tokens 512

# 3. Re-score a trained model (4 samples per problem halves the sampling noise):
python zero_shot_eval.py -m output/checkpoints/<run>_<timestamp> --max_tokens 512 --n_samples 4
```

- **Useful flags:** `--n_grpo_steps`, `--eval_every`, `--seed`, `--lr`, `--run_tag`, `--no-save_model`.
- **Outputs:**
  - TensorBoard logs and `train_metrics.jsonl` / `eval_metrics.jsonl` / `config.json` go to `output/tb/<loss>_L<len>[_tag]/<timestamp>/`.
  - Models go to `output/checkpoints/` (3.3 GB each).

## Rollouts

`train_logged.py` takes the same flags as `starter.py` and also saves every rollout, so you can read what the model wrote at each step.

```bash
python train_logged.py --loss_type grpo --max_tokens 256
python serve.py                  # then open http://127.0.0.1:44834/rollouts.html
```

- **Files:** `output/rollouts/<run>/<timestamp>/train/step_NNNN.jsonl` holds all 128 rollouts of a step with their group, reward and advantage. `eval/step_NNNN.jsonl` holds one sample per test problem. Each line is one completion with its tokens and log-probs.
- **Viewer:** a table of every saved step (accuracy, 0.1 and 0.0 counts, truncations, groups with no learning signal), then the rollouts of a step, then one rollout with its tokens shaded by surprise. Click a cell to filter on it, click a header to sort, and use <kbd>j</kbd> / <kbd>k</kbd> to step through rollouts.
- `starter.py` is not modified. `train_logged.py` wraps its functions and then calls `starter.main()`.

## Modal

`run_modal.py` runs any script here on a Modal H200 and keeps `output/` on the `grpo-output` Volume.

```bash
uvx modal run run_modal.py --args "zero_shot_eval.py --max_tokens 512"
uvx modal run run_modal.py --args "train_logged.py --loss_type grpo --max_tokens 256"
STARTER=path/to/your/starter.py uvx modal run run_modal.py --args "..."   # upload another starter.py instead

uvx modal volume get --force grpo-output rollouts output/   # copy the rollouts here, then: python serve.py
```

## Acknowledgments

We are grateful to [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for the Countdown task and to the Stanford CS336 staff for Assignment 5, which served as a basis for this project. The rollout engine builds on Jonathan Chang's [flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm) (MIT License; see `THIRD_PARTY_NOTICES.md`).
