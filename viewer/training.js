// The training tab: the runs the assignment asks for, one row each. The zero-shot baselines sit beside the GRPO and
// Dr.GRPO runs so each trained model reads against the base model at its length. A trained run opens its steps
// (?run=), what each step's rollouts scored beside what the update did with them; a step opens a table of its 128
// rollouts (?run=&step=), and a rollout opens its tokens (?run=&step=&rollout=).
// A run too short to compare (a smoke test) is left out; it is still on the rollouts tab.

import {
  addInfoField, createTable, fatal, fetchJsonl, formatAdvantage, formatNumber, loadIndex, newestByRun, percent, summaryEl,
} from "./common.js";
import {openRollout} from "./rollout-view.js";

const MIN_STEPS = 10;

const COLUMNS = [
  {key: "run", label: "Run"},
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

const STEP_COLUMNS = [
  {key: "step", label: "Step"},
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

// One row per rollout of a step, in group order, so a group's eight samples and their advantages read together.
const ROLLOUT_COLUMNS = [
  {key: "index", label: "#"},
  {key: "group", label: "Group", num: true, filter: true},
  {key: "sample", label: "Sample", num: true},
  {key: "target", label: "Target", num: true},
  {key: "numbers", label: "Numbers", value: (r) => r.numbers.join(", ")},
  {key: "equation", label: "Answer", value: (r) => r.equation ?? "", text: (r) => r.equation ?? "—"},
  {key: "result", label: "Value", num: true, value: (r) => (r.result === null ? NaN : r.result),
   text: (r) => (r.result === null ? "—" : String(+r.result.toFixed(4)))},
  {key: "reward", label: "Reward", num: true, filter: true, text: (r) => rewardText(r.reward)},
  {key: "advantage", label: "Advantage", num: true, text: (r) => formatAdvantage(r.advantage)},
  {key: "output_tokens", label: "Tokens", num: true},
  {key: "finish_reason", label: "Finish", filter: true, value: (r) => (r.finish_reason === "length" ? "truncated" : "stopped")},
];

const METHODS = {grpo: "grpo", dr_grpo: "dr-grpo", grpo_no_len_norm: "grpo-no-len-norm"};
// What a run's training reward was, by its run tag (starter_lenpen.py); every other run kept the handout's.
const REWARD_SHAPES = {lenpen: "1 − 1e-6·tokens / 0 / 0", group_lenpen: "1 − 1e-6·tokens / 0 / 0"};
const BASE_REWARD = "1 / 0.1 / 0";
// What the loss normalizes: GRPO divides each rollout's loss by its length and each advantage by its group's reward
// std; Dr.GRPO does neither.
const NORMED = {grpo: "length, advantage", dr_grpo: "none", grpo_no_len_norm: "advantage"};

// 1.0 / 0.1 / 0 read as one decimal; a length-shaped reward keeps the digits that carry the penalty.
function rewardText(reward) {
  return Number.isInteger(Math.round(reward * 1e6) / 1e5) ? reward.toFixed(1) : reward.toFixed(6);
}

function summarize(lines) {
  summaryEl.replaceChildren();
  for (const [text, cls] of lines) {
    const line = document.createElement("div");
    line.textContent = text;
    if (cls) line.className = cls;
    summaryEl.append(line);
  }
}

async function showRollouts(run, entry) {
  const rows = await fetchJsonl(entry.path);
  addInfoField("run", run.run);
  addInfoField("method", `${METHODS[run.config.loss_type] ?? run.config.loss_type} · L${run.config.max_tokens}`);
  addInfoField("step", String(entry.step));
  const table = createTable({
    head: document.querySelector("#table-head"),
    body: document.querySelector("#table-body"),
    columns: ROLLOUT_COLUMNS,
    defaultSort: [{key: "index"}],
    onOpen: (row) => {
      location.href = `training.html?${new URLSearchParams({run: run.run, step: String(entry.step), rollout: String(row.index)})}`;
    },
    onRender: (shown) => summarize([
      [`${formatNumber(shown.length)} rollouts`, ""],
      [`${formatNumber(percent(shown.filter((r) => r.reward > 0.5).length, shown.length), 1)}% correct`, ""],
      [`${formatNumber(shown.reduce((n, r) => n + r.output_tokens, 0))} generated`, "completion-count"],
    ]),
  });
  return {table, rows};
}

async function showSteps(run, stepsIndex) {
  const [train, evals] = await Promise.all([
    fetchJsonl(`${run.base}/train_metrics.jsonl`).catch(() => []),
    fetchJsonl(`${run.base}/eval_metrics.jsonl`).catch(() => []),
  ]);
  const metrics = new Map(train.map((r) => [r.step, r]));
  const evalAt = new Map(evals.map((r) => [r.step, r.accuracy]));
  const rows = stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts && e.split === "train")
    .map((e) => ({...metrics.get(e.step), ...e, eval: evalAt.get(e.step)}));
  addInfoField("run", run.run);
  addInfoField("method", `${METHODS[run.config.loss_type] ?? run.config.loss_type} · L${run.config.max_tokens}`);
  const table = createTable({
    head: document.querySelector("#table-head"),
    body: document.querySelector("#table-body"),
    columns: STEP_COLUMNS,
    defaultSort: [{key: "step"}],
    onOpen: (row) => {
      location.href = `training.html?${new URLSearchParams({run: row.run, step: String(row.step)})}`;
    },
    onRender: (shown) => summarize([
      [`${formatNumber(shown.length)} steps`, ""],
      [`${formatNumber(shown.reduce((n, r) => n + r.count, 0))} rollouts`, ""],
      [`${formatNumber(shown.reduce((n, r) => n + r.completion_tokens, 0))} generated`, "completion-count"],
    ]),
  });
  return {table, rows};
}

try {
  const [runsIndex, stepsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
  ]);
  const runs = runsIndex.filter((r) => r.type === "run");
  const newest = newestByRun(runs);
  const query = new URLSearchParams(location.search);
  const open = runs.find((r) => r.run === query.get("run") && newest.get(r.run) === r.ts);
  const step = open && stepsIndex.find((e) =>
    e.run === open.run && e.ts === open.ts && e.split === "train" && String(e.step) === query.get("step"));
  if (open && step && query.get("rollout")) {
    document.querySelector("#loading-state").style.display = "none";
    await openRollout({
      entry: step, rows: await fetchJsonl(step.path),
      method: `${METHODS[open.config.loss_type] ?? open.config.loss_type} · L${open.config.max_tokens}`,
      onClose: () => { location.href = `training.html?${new URLSearchParams({run: open.run, step: String(step.step)})}`; },
    });
  } else if (open) {
    const {table, rows} = step ? await showRollouts(open, step) : await showSteps(open, stepsIndex);
    document.querySelector("#loading-state").style.display = "none";
    table.setRows(rows);
  } else {
  const trained = await Promise.all(runs
    .filter((r) => newest.get(r.run) === r.ts && r.config.n_grpo_steps >= MIN_STEPS)
    .map(async (run) => {
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
      if (row.steps) location.href = `training.html?${new URLSearchParams({run: row.run})}`;
    },
    onRender: (shown) => summarize([[`${shown.length} runs`], [`${shown.filter((r) => r.steps).length} trained`],
      [`${shown.filter((r) => !r.steps).length} zero-shot`]]),
  });
  document.querySelector("#loading-state").style.display = "none";
  if (!rows.length) document.querySelector("#empty-state").style.display = "block";
  table.setRows(rows);
  }
} catch (error) {
  fatal(error);
}
