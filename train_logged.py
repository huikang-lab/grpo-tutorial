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

LOGPROB_DIGITS = 3


class RolloutLogger:
    def __init__(self) -> None:
        self.root: str | None = None
        self.tokenizer = None
        self.step = 0
        self.last_results: List[Any] = []
        self.pending: List[Any] = []

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
        return {
            **extra, "target": ground_truth["target"], "numbers": ground_truth["numbers"], "reward": float(reward),
            "equation": equation, "result": result, "finish_reason": out.finish_reason,
            "output_tokens": len(out.token_ids), "prompt": prompt, "text": out.text, "tokens": list(out.token_ids),
            "pieces": self.tokenizer.batch_decode([[t] for t in out.token_ids]), "logprobs": logprobs,
        }


logger = RolloutLogger()


def patch_starter() -> None:
    init_policy, generate, summary_writer = starter.init_policy, starter.FlexLLM.generate, starter.SummaryWriter
    rollout_with_engine, compute_advantages = starter.rollout_with_engine, starter.compute_group_normalized_advantages
    evaluate_model, log_train = starter.evaluate_model, starter.log_train

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
            logger.write("train", logger.step + 1, [
                logger.record(prompt, out, gt, raw_rewards[i].item(), index=i, group=i // group_size,
                              sample=i % group_size, advantage=round(advantages[i].item(), 4))
                for i, ((prompt, out), gt) in enumerate(zip(logger.pending, ground_truths))])
        logger.pending = []
        return advantages, raw_rewards, metadata

    def logged_evaluate_model(llm, sampling_params, eval_prompts, eval_answers):
        metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
        logger.write("eval", logger.step, [
            logger.record(result.prompt, result.outputs[0], gt, example["reward"], index=i)
            for i, (result, gt, example) in enumerate(zip(logger.last_results, eval_answers, metrics["examples"]))])
        return metrics

    def logged_log_train(*args, **kwargs):
        log_train(*args, **kwargs)
        logger.step += 1

    starter.init_policy = logged_init_policy
    starter.FlexLLM.generate = logged_generate
    starter.SummaryWriter = LoggedSummaryWriter
    starter.rollout_with_engine = logged_rollout_with_engine
    starter.compute_group_normalized_advantages = logged_compute_advantages
    starter.evaluate_model = logged_evaluate_model
    starter.log_train = logged_log_train


if __name__ == "__main__":
    patch_starter()
    starter.main()
