"""
Train with starter.py and save every rollout (training groups and eval samples) as JSONL for the rollout viewer.

    python train_logged.py --loss_type grpo --max_tokens 256      # same flags as starter.py

Rollouts go to output/rollouts/<run>/<timestamp>/{train,eval}/step_NNNN.jsonl, one line per completion, next to the
TensorBoard logs in output/tb/<run>/<timestamp>/. starter.py is not modified: its module-level functions are wrapped
before starter.main() runs, so the same file works with any implementation of Tasks 1-5.
"""
import inspect
import json
import os
import sys
from typing import Any, Dict, List

import starter

LOGPROB_DIGITS = 6


class RolloutLogger:
    def __init__(self) -> None:
        self.root: str | None = None
        self.tokenizer = None
        self.step = 0
        self.last_results: List[Any] = []
        self.pending: List[Any] = []
        self.records: List[Dict[str, Any]] = []  # this step's train records, written once training has scored them
        self.last_log_probs = None
        self.train_rows: List[Any] = []  # (label ids, log-probs) of each response, in rollout order

    def write(self, kind: str, step: int, records: List[Dict[str, Any]]) -> None:
        if self.root is None:
            return
        path = os.path.join(self.root, kind, f"step_{step:04d}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(path + ".tmp", path)  # the viewer never sees a half-written step

    def record(self, prompt: str, out: Any, ground_truth: Dict, reward: float, **extra: Any) -> Dict[str, Any]:
        equation = starter._extract_answer(out.text)
        result = starter._evaluate_equation(equation) if equation is not None else None
        logprobs = None
        if out.logprobs:
            logprobs = [round(next(iter(lp.values())).logprob, LOGPROB_DIGITS) for lp in out.logprobs]
        # Only ids are kept: the viewer decodes them with tokenizer/tokenizer.json. A group shares one prompt, so it
        # rides on sample 0 alone.
        prompt_tokens = self.tokenizer(prompt)["input_ids"] if extra.get("sample", 0) == 0 else None
        return {
            **extra, "target": ground_truth["target"], "numbers": ground_truth["numbers"], "reward": float(reward),
            "equation": equation, "result": result, "finish_reason": out.finish_reason,
            **({"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {}),
            "output_tokens": len(out.token_ids), "tokens": list(out.token_ids), "logprobs": logprobs,
        }


logger = RolloutLogger()


def patch_starter() -> None:
    init_policy, generate, summary_writer = starter.init_policy, starter.FlexLLM.generate, starter.SummaryWriter
    rollout_with_engine, compute_advantages = starter.rollout_with_engine, starter.compute_group_normalized_advantages
    evaluate_model, log_train = starter.evaluate_model, starter.log_train
    get_response_log_probs, microbatch_step = starter.get_response_log_probs, starter.grpo_microbatch_step

    def logged_init_policy(*args, **kwargs):
        policy, tokenizer = init_policy(*args, **kwargs)
        logger.tokenizer = tokenizer
        return policy, tokenizer

    def logged_generate(self, *args, **kwargs):
        logger.last_results = generate(self, *args, **kwargs)
        return logger.last_results

    class LoggedSummaryWriter(summary_writer):
        def __init__(self, *args, log_dir: str, **kwargs):
            super().__init__(*args, log_dir=log_dir, **kwargs)
            run, timestamp = os.path.normpath(log_dir).split(os.sep)[-2:]
            logger.root = os.path.join("output", "rollouts", run, timestamp)
            os.makedirs(logger.root, exist_ok=True)
            with open(os.path.join(logger.root, "run.json"), "w") as f:
                json.dump({"run": run, "timestamp": int(timestamp), "argv": sys.argv[1:], "tb_dir": log_dir}, f, indent=1)

    def logged_rollout_with_engine(*args, **kwargs):
        outputs = rollout_with_engine(*args, **kwargs)
        logger.pending = [(result.prompt, out) for result in logger.last_results for out in result.outputs]
        return outputs

    def logged_compute_advantages(*args, **kwargs):
        advantages, raw_rewards, metadata = compute_advantages(*args, **kwargs)
        bound = inspect.signature(compute_advantages).bind(*args, **kwargs).arguments
        ground_truths, group_size = bound["repeated_ground_truths"], bound["group_size"]
        if len(logger.pending) == len(ground_truths):
            logger.records = [
                logger.record(prompt, out, gt, raw_rewards[i].item(), index=i, group=i // group_size,
                              sample=i % group_size, advantage=round(advantages[i].item(), 4))
                for i, ((prompt, out), gt) in enumerate(zip(logger.pending, ground_truths))]
        logger.pending = []
        logger.train_rows = []
        return advantages, raw_rewards, metadata

    def logged_get_response_log_probs(*args, **kwargs):
        logger.last_log_probs = get_response_log_probs(*args, **kwargs)
        return logger.last_log_probs

    def logged_microbatch_step(*args, **kwargs):
        result = microbatch_step(*args, **kwargs)
        bound = inspect.signature(microbatch_step).bind(*args, **kwargs).arguments
        labels, mask, log_probs = bound["labels"], bound["response_mask"], logger.last_log_probs.detach().float()
        for row in range(labels.shape[0]):
            keep = mask[row]
            logger.train_rows.append((labels[row][keep].tolist(), log_probs[row][keep].tolist()))
        return result

    def attach_train_log_probs() -> None:
        # The trainer re-tokenizes the response text, so its ids can differ from the sampled ones. Log-probs are kept
        # per sampled token only when the ids match; otherwise the trainer's own ids are kept beside them.
        if len(logger.train_rows) != len(logger.records):
            return
        for record, (ids, log_probs) in zip(logger.records, logger.train_rows):
            record["train_logprobs"] = [round(x, LOGPROB_DIGITS) for x in log_probs]
            if ids != record["tokens"]:
                record["train_tokens"] = ids

    def logged_evaluate_model(llm, sampling_params, eval_prompts, eval_answers):
        metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
        logger.write("eval", logger.step, [
            logger.record(result.prompt, result.outputs[0], gt, example["reward"], index=i)
            for i, (result, gt, example) in enumerate(zip(logger.last_results, eval_answers, metrics["examples"]))])
        return metrics

    def logged_log_train(*args, **kwargs):
        log_train(*args, **kwargs)
        logger.step += 1
        if logger.records:
            attach_train_log_probs()
            logger.write("train", logger.step, logger.records)
        logger.records, logger.train_rows = [], []

    starter.init_policy = logged_init_policy
    starter.FlexLLM.generate = logged_generate
    starter.SummaryWriter = LoggedSummaryWriter
    starter.rollout_with_engine = logged_rollout_with_engine
    starter.compute_group_normalized_advantages = logged_compute_advantages
    starter.evaluate_model = logged_evaluate_model
    starter.log_train = logged_log_train
    starter.get_response_log_probs = logged_get_response_log_probs
    starter.grpo_microbatch_step = logged_microbatch_step


if __name__ == "__main__":
    patch_starter()
    starter.main()
