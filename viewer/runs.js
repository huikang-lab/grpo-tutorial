// The runs tab: the runs the assignment asks for, one row each. The zero-shot baselines sit beside the GRPO and
// Dr.GRPO runs so each trained model reads against the base model at its length. A trained run opens its steps.

import {
  createTable, fatal, fetchJsonl, formatNumber, hideLoading, loadIndex, METHODS, showEmpty, summarize, trainedRuns,
} from "./common.js";

const COLUMNS = [
  {key: "run", label: "Run", render: runLink},
  {key: "method", label: "Method", filter: true},
  {key: "reward_shape", label: "Reward shape", filter: true,
   title: "training reward for correct / wrong answer / no answer; eval always scores 1 / 0.1 / 0"},
  {key: "normed", label: "Normed", filter: true,
   title: "length: each rollout's loss divided by its own token count; advantage: divided by the group's reward std"},
  {key: "max_tokens", label: "Max tokens", num: true, filter: true},
  {key: "steps", label: "Steps", num: true, text: (r) => (r.steps ? String(r.steps) : "—")},
  {key: "start", label: "Step 0", num: true, text: (r) => (Number.isFinite(r.start) ? `${formatNumber(r.start, 1)}%` : "—"),
   title: "eval accuracy before training"},
  {key: "final", label: "Final eval", num: true, text: (r) => `${formatNumber(r.final, 1)}%`,
   title: "eval accuracy at the last step: 1024 test problems, one sample each"},
  {key: "best", label: "Best eval", num: true, text: (r) => `${formatNumber(r.best, 1)}%`},
  {key: "avg_tokens", label: "Completion", num: true, text: (r) => formatNumber(r.avg_tokens, 0),
   title: "mean tokens per eval answer at the last step"},
  {key: "truncated", label: "Truncated", num: true, text: (r) => `${formatNumber(r.truncated, 1)}%`,
   title: "eval answers stopped at max_tokens at the last step"},
  {key: "generated", label: "Generated", num: true, text: (r) => formatNumber(r.generated),
   title: "every token the run generated: training rollouts and eval answers (zero-shot: its one eval)"},
  {key: "ts", label: "Started", num: true, text: (r) => (r.ts ? startedText(r.ts) : "—"), title: "when the run began, local time"},
  {key: "train_seconds", label: "Train time", num: true, text: (r) => durationText(r.train_seconds),
   title: "generating plus training, summed over the steps; model loading and evals are not logged"},
];

// A trained run's name links to its steps; a zero-shot baseline has none to show.
function runLink(row) {
  if (!row.steps) return document.createTextNode(row.run);
  const link = document.createElement("a");
  link.href = `steps.html?${new URLSearchParams({run: row.run})}`;
  link.textContent = row.run;
  return link;
}

function startedText(ts) {
  const date = new Date(ts * 1000);
  const time = date.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
  return `${date.toLocaleDateString(undefined, {month: "short", day: "numeric"})} ${time}`;
}

function durationText(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const minutes = Math.round(seconds / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)}h ${minutes % 60}m` : `${minutes}m`;
}

// What a run's training reward was, by its run tag (starter_lenpen.py); every other run kept the handout's.
const REWARD_SHAPES = {lenpen: "1 − 1e-6·tokens / 0 / 0", group_lenpen: "1 − 1e-6·tokens / 0 / 0"};
const BASE_REWARD = "1 / 0.1 / 0";
// What the loss normalizes: GRPO divides each rollout's loss by its length and each advantage by its group's reward
// std; Dr.GRPO does neither.
const NORMED = {grpo: "length, advantage", dr_grpo: "none", grpo_no_len_norm: "advantage"};

try {
  const [runsIndex, stepsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
  ]);
  const trained = await Promise.all(trainedRuns(runsIndex).map(async (run) => {
    const [evals, train] = await Promise.all([
      fetchJsonl(`${run.base}/eval_metrics.jsonl`).catch(() => []),
      fetchJsonl(`${run.base}/train_metrics.jsonl`).catch(() => []),
    ]);
    const last = stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts && e.split === "eval")
      .sort((a, b) => a.step - b.step).at(-1);
    return {
      run: run.run, ts: run.ts, method: METHODS[run.config.loss_type] ?? run.config.loss_type,
      reward_shape: REWARD_SHAPES[run.config.run_tag] ?? BASE_REWARD, normed: NORMED[run.config.loss_type] ?? "—",
      max_tokens: run.config.max_tokens,
      steps: stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts && e.split === "train").length,
      start: evals[0]?.accuracy, final: evals.at(-1)?.accuracy, best: Math.max(...evals.map((r) => r.accuracy)),
      avg_tokens: evals.at(-1)?.avg_output_tokens, truncated: last ? (100 * last.truncated) / last.count : NaN,
      generated: stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts)
        .reduce((n, e) => n + e.completion_tokens, 0),
      train_seconds: train.length ? train.reduce((n, r) => n + r.t_gen + r.t_train, 0) : NaN,
    };
  }));
  const baselines = runsIndex.filter((r) => r.type === "zero_shot").map((z) => ({
    run: `zero_shot_L${z.max_tokens}`, method: "zero-shot", reward_shape: BASE_REWARD, normed: "—",
    max_tokens: z.max_tokens, steps: 0, start: NaN,
    final: z.accuracy, best: z.accuracy, avg_tokens: z.avg_tokens, truncated: 100 * z.truncated_frac,
    generated: Math.round(z.avg_tokens * z.n_problems * z.n_samples), ts: z.ts ?? null, train_seconds: NaN,
  }));
  const rows = [...baselines, ...trained];
  const table = createTable({
    head: document.querySelector("#table-head"),
    body: document.querySelector("#table-body"),
    columns: COLUMNS,
    // In the order the runs were started, ties by length.
    defaultSort: [{key: "ts"}, {key: "max_tokens"}],
    onOpen: (row) => {
      if (row.steps) location.href = `steps.html?${new URLSearchParams({run: row.run})}`;
    },
    onRender: (shown) => summarize([[`${shown.length} runs`], [`${shown.filter((r) => r.steps).length} trained`],
      [`${shown.filter((r) => !r.steps).length} zero-shot`]]),
  });
  hideLoading();
  if (!rows.length) showEmpty();
  table.setRows(rows);
} catch (error) {
  fatal(error);
}
