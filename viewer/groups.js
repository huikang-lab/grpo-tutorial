// The groups tab: every group of every training step, the prompt's eight attempts summarized in one row, newest step
// first, with one cell per attempt's reward. A group whose attempts all scored the same has zero advantage and teaches
// nothing. ?run= and ?step= narrow it; a group opens its attempts, a reward cell that one attempt.

import {
  addInfoField, createTable, fatal, formatNumber, hideLoading, loadIndex, methodText, percent, showEmpty, summarize,
  trainedRuns,
} from "./common.js";

const COLUMNS = [
  {key: "run", label: "Run", filter: true},
  {key: "step", label: "Step", num: true, filter: true},
  {key: "group", label: "Group", num: true},
  {key: "target", label: "Target", num: true},
  {key: "numbers", label: "Numbers", value: (r) => r.numbers.join(", ")},
  {key: "rewards", label: "Rewards", value: (r) => r.mean_reward, render: rewardCells,
   title: "each attempt's reward in sample order: green correct, amber answered but wrong, grey no answer; " +
     "hover for whether it was truncated at max_tokens. Click one to open it."},
  {key: "mean_reward", label: "Reward", num: true, text: (r) => formatNumber(r.mean_reward, 3), title: "group mean"},
  {key: "reward_std", label: "Reward std", num: true, text: (r) => formatNumber(r.reward_std, 3)},
  {key: "flat", label: "Zero-adv", filter: true, value: (r) => (r.flat ? "yes" : "no"),
   title: "every attempt scored the same, so every advantage is zero"},
  {key: "completion_tokens", label: "Generated", num: true, text: (r) => formatNumber(r.completion_tokens)},
];

// The handout's three rewards, and a length-shaped one (just under 1.0 when correct) read as correct.
function rewardClass(reward) {
  if (reward > 0.5) return "correct";
  if (reward > 0.05) return "partial";
  return "none";
}

function rewardCells(row) {
  const cells = document.createElement("div");
  cells.className = "reward-cells";
  row.rewards.forEach((reward, sample) => {
    const cell = document.createElement("a");
    cell.className = `reward-cell ${rewardClass(reward)}${row.truncations[sample] ? " truncated" : ""}`;
    cell.textContent = {correct: "1", partial: ".1", none: "0"}[rewardClass(reward)];
    cell.title = `sample ${sample} · reward ${+reward.toFixed(6)}${row.truncations[sample] ? " · truncated" : ""}`;
    cell.href = `attempts.html?${new URLSearchParams({run: row.run, step: String(row.step), group: String(row.group),
      rollout: String(row.group * row.count + sample)})}`;
    // The row beneath opens the whole group, so a cell click must not also open that.
    cell.addEventListener("click", (event) => event.stopPropagation());
    cells.append(cell);
  });
  return cells;
}

try {
  const [runsIndex, groupsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("groups-index.jsonl", "grpo-groups-index-v3"),
  ]);
  const runs = trainedRuns(runsIndex);
  const kept = new Set(runs.map((r) => `${r.run}/${r.ts}`));
  const rows = groupsIndex.filter((g) => kept.has(`${g.run}/${g.ts}`));
  const table = createTable({
    head: document.querySelector("#table-head"),
    body: document.querySelector("#table-body"),
    columns: COLUMNS,
    // Newest first: the latest run's last step on top, its groups in order.
    defaultSort: [{key: "ts", desc: true}, {key: "step", desc: true}, {key: "group"}],
    pageSize: 1000,
    onOpen: (row) => {
      location.href = `attempts.html?${new URLSearchParams({run: row.run, step: String(row.step), group: String(row.group)})}`;
    },
    onRender: (shown) => {
      const attempts = shown.reduce((n, r) => n + r.count, 0);
      summarize([
        [`${formatNumber(shown.length)} groups`, ""],
        [`${formatNumber(shown.filter((r) => r.flat).length)} zero-adv`, ""],
        [`${formatNumber(percent(shown.reduce((n, r) => n + r.correct, 0), attempts), 1)}% correct`, ""],
        [`${formatNumber(shown.reduce((n, r) => n + r.completion_tokens, 0))} generated`, "completion-count"],
      ]);
    },
  });
  const open = runs.find((r) => r.run === table.filters.get("run"));
  if (open) {
    addInfoField("run", open.run);
    addInfoField("method", methodText(open));
  }
  if (table.filters.has("step")) addInfoField("step", table.filters.get("step"));
  hideLoading();
  if (!rows.length) showEmpty();
  table.setRows(rows);
} catch (error) {
  fatal(error);
}
