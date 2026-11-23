// The steps tab: every training step of every run, what its rollouts scored beside what the update did with them.
// ?run= keeps one run (a * wildcard keeps several); a step opens its groups.

import {
  addInfoField, createTable, fatal, fetchJsonl, formatNumber, hideLoading, loadIndex, methodText, percent, showEmpty,
  summarize, trainedRuns,
} from "./common.js";

const COLUMNS = [
  {key: "run", label: "Run", filter: true},
  {key: "step", label: "Step", num: true},
  {key: "reward_mean", label: "Reward", num: true, text: (r) => formatNumber(r.reward_mean, 3),
   title: "mean over the step's 128 rollouts"},
  {key: "accuracy", label: "Accuracy", num: true, value: (r) => percent(r.correct, r.count),
   text: (r) => `${formatNumber(percent(r.correct, r.count), 1)}%`, title: "reward 1.0"},
  {key: "flat_groups", label: "Zero-adv groups", num: true, text: (r) => `${r.flat_groups} of ${r.groups}`,
   title: "groups whose rollouts all scored the same, so they teach nothing"},
  {key: "truncated", label: "Truncated", num: true, title: "stopped at max_tokens"},
  {key: "completion_tokens", label: "Generated", num: true, text: (r) => formatNumber(r.completion_tokens)},
  {key: "loss", label: "Loss", num: true,
   text: (r) => (Number.isFinite(r.loss) ? (r.loss === 0 ? "0" : r.loss.toExponential(2)) : "—")},
  {key: "grad_norm", label: "Grad norm", num: true, text: (r) => formatNumber(r.grad_norm, 4), title: "before clipping to 1.0"},
  {key: "eval", label: "Eval", num: true, text: (r) => (Number.isFinite(r.eval) ? `${formatNumber(r.eval, 1)}%` : ""),
   title: "eval accuracy after this step, where one ran"},
  {key: "logprob_diff", label: "Logprob diff", num: true, text: (r) => formatNumber(r.logprob_diff, 4),
   title: "training minus rollout log-prob, per token"},
];

try {
  const [runsIndex, stepsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
  ]);
  const runs = trainedRuns(runsIndex);
  const rows = (await Promise.all(runs.map(async (run) => {
    const [train, evals] = await Promise.all([
      fetchJsonl(`${run.base}/train_metrics.jsonl`).catch(() => []),
      fetchJsonl(`${run.base}/eval_metrics.jsonl`).catch(() => []),
    ]);
    const metrics = new Map(train.map((r) => [r.step, r]));
    const evalAt = new Map(evals.map((r) => [r.step, r.accuracy]));
    return stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts && e.split === "train")
      .map((e) => ({...metrics.get(e.step), ...e, eval: evalAt.get(e.step)}));
  }))).flat();
  const table = createTable({
    head: document.querySelector("#table-head"),
    body: document.querySelector("#table-body"),
    columns: COLUMNS,
    defaultSort: [{key: "run"}, {key: "step"}],
    onOpen: (row) => {
      location.href = `groups.html?${new URLSearchParams({run: row.run, step: String(row.step)})}`;
    },
    onRender: (shown) => summarize([
      [`${formatNumber(shown.length)} steps`, ""],
      [`${formatNumber(shown.reduce((n, r) => n + r.count, 0))} rollouts`, ""],
      [`${formatNumber(shown.reduce((n, r) => n + r.completion_tokens, 0))} generated`, "completion-count"],
    ]),
  });
  const open = runs.find((r) => r.run === table.filters.get("run"));
  if (open) {
    addInfoField("run", open.run);
    addInfoField("method", methodText(open));
  }
  hideLoading();
  if (!rows.length) showEmpty();
  table.setRows(rows);
} catch (error) {
  fatal(error);
}
