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
./serve.sh                       # prints the URLs: runs, metrics, steps, groups, attempts
```

- **Files:** `output/rollouts/<run>/<timestamp>/train/step_NNNN.jsonl` holds all 128 rollouts of a step with their group, reward and advantage. `eval/step_NNNN.jsonl` holds one sample per test problem. Each line is one completion with its prompt and completion tokens, the rollout log-prob of each completion token and, for training rollouts, the trainer's log-prob of it.
- **Runs:** the trained runs and the zero-shot baselines, one row each. A trained run opens its steps.
- **Steps:** every training step of every run: reward, zero-advantage groups, truncations, loss, grad norm, eval, log-prob diff. A step opens its groups.
- **Groups:** every group of every step: the prompt, how many of its eight attempts were correct, answered or truncated, and whether it had zero advantage. A group opens its attempts.
- **Attempts:** the 128 rollouts of one step, filterable to one group. An attempt opens its prompt and completion as one token stream, shaded by surprise or by training minus rollout log-prob, beside its reward and its group.
- On every table, click a blue value to filter on it and a header to sort; the filters live in the URL. In an attempt, <kbd>j</kbd> / <kbd>k</kbd> step through the attempts and <kbd>Esc</kbd> goes back.
- **Metrics:** every run on one set of charts, with the zero-shot baselines from `output/zero_shot/L<len>.json` as dashed lines; pick a run for its own charts and config.
- `./serve.sh` serves in the background, keeps `rollouts-index.jsonl` and `runs-index.jsonl` fresh, and replaces an earlier server on the same port. Logs go to `logs/`.
- `starter.py` is not modified. `train_logged.py` wraps its functions and then calls `starter.main()`.

## Modal

`run_modal.py` runs any script here on a Modal H200 and keeps `output/` on the `grpo-output` Volume.

```bash
uvx modal run run_modal.py --args "zero_shot_eval.py --max_tokens 512"
uvx modal run run_modal.py --args "train_logged.py --loss_type grpo --max_tokens 256"
STARTER=path/to/your/starter.py uvx modal run run_modal.py --args "..."   # upload another starter.py instead

mkdir -p output && uvx modal volume get --force grpo-output rollouts output/   # copy the rollouts here
uvx modal volume get --force grpo-output tb output/                            # and the metrics, then: ./serve.sh
```

## Acknowledgments

We are grateful to [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for the Countdown task and to the Stanford CS336 staff for Assignment 5, which served as a basis for this project. The rollout engine builds on Jonathan Chang's [flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm) (MIT License; see `THIRD_PARTY_NOTICES.md`).
