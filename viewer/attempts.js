// The attempts tab: one row per rollout of a training step (?run=&step=), in group order, so a group's eight attempts
// and their advantages read together; ?group= keeps one group. An attempt opens its tokens (&rollout=).
// A step's file holds every token of its 128 attempts, so the tab loads one step at a time.

import {
  addInfoField, createTable, fatal, fetchJsonl, formatAdvantage, formatNumber, hideLoading, loadIndex, methodText,
  percent, showEmpty, summarize, trainedRuns,
} from "./common.js";
import {openRollout} from "./rollout-view.js";

const COLUMNS = [
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

// 1.0 / 0.1 / 0 read as one decimal; a length-shaped reward keeps the digits that carry the penalty.
function rewardText(reward) {
  return Number.isInteger(Math.round(reward * 1e6) / 1e5) ? reward.toFixed(1) : reward.toFixed(6);
}

function stepUrl(run, step) {
  const query = new URLSearchParams({run, step: String(step)});
  const group = new URLSearchParams(location.search).get("group");
  if (group) query.set("group", group);
  return `attempts.html?${query}`;
}

try {
  const [runsIndex, stepsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
  ]);
  const query = new URLSearchParams(location.search);
  const open = trainedRuns(runsIndex).find((r) => r.run === query.get("run"));
  const entry = open && stepsIndex.find((e) =>
    e.run === open.run && e.ts === open.ts && e.split === "train" && String(e.step) === query.get("step"));
  if (!entry) {
    hideLoading();
    showEmpty(query.get("run") ? `No training step ${query.get("step") ?? ""} in ${query.get("run")}.` : undefined);
  } else if (query.get("rollout")) {
    hideLoading();
    await openRollout({
      entry, rows: await fetchJsonl(entry.path), method: methodText(open),
      onClose: () => { location.href = stepUrl(open.run, entry.step); },
    });
  } else {
    const rows = await fetchJsonl(entry.path);
    addInfoField("run", open.run);
    addInfoField("method", methodText(open));
    addInfoField("step", String(entry.step));
    const table = createTable({
      head: document.querySelector("#table-head"),
      body: document.querySelector("#table-body"),
      columns: COLUMNS,
      defaultSort: [{key: "index"}],
      onOpen: (row) => {
        location.href = `attempts.html?${new URLSearchParams({...Object.fromEntries(new URLSearchParams(location.search)),
          rollout: String(row.index)})}`;
      },
      onRender: (shown) => summarize([
        [`${formatNumber(shown.length)} attempts`, ""],
        [`${formatNumber(percent(shown.filter((r) => r.reward > 0.5).length, shown.length), 1)}% correct`, ""],
        [`${formatNumber(shown.reduce((n, r) => n + r.output_tokens, 0))} generated`, "completion-count"],
      ]),
    });
    if (table.filters.has("group")) addInfoField("group", table.filters.get("group"));
    hideLoading();
    table.setRows(rows);
  }
} catch (error) {
  fatal(error);
}
