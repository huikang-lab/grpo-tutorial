// The attempts tab: every rollout of every training step, newest step first, a group's eight attempts together with
// their advantages. ?run=, ?step= and ?group= narrow it; an attempt opens its tokens (&rollout=), which come from its
// step's file. The table itself reads groups-index.jsonl, so it never fetches the step files.

import {
  addInfoField, createTable, fatal, fetchJsonl, formatAdvantage, formatNumber, hideLoading, loadIndex, methodText,
  percent, showEmpty, summarize, trainedRuns,
} from "./common.js";
import {openRollout} from "./rollout-view.js";

const COLUMNS = [
  {key: "run", label: "Run", filter: true},
  {key: "step", label: "Step", num: true, filter: true},
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
  {key: "finish", label: "Finish", filter: true},
];

// 1.0 / 0.1 / 0 read as one decimal; a length-shaped reward keeps the digits that carry the penalty.
function rewardText(reward) {
  return Number.isInteger(Math.round(reward * 1e6) / 1e5) ? reward.toFixed(1) : reward.toFixed(6);
}

// One row per attempt, unpacked from its group's line in the index.
function attemptsOf(group) {
  return group.rewards.map((reward, sample) => ({
    run: group.run, ts: group.ts, step: group.step, group: group.group, sample,
    index: group.group * group.count + sample, target: group.target, numbers: group.numbers,
    equation: group.equations[sample], result: group.results[sample], reward, advantage: group.advantages[sample],
    output_tokens: group.output_tokens[sample], finish: group.truncations[sample] ? "truncated" : "stopped",
  }));
}

try {
  const [runsIndex, stepsIndex, groupsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
    loadIndex("groups-index.jsonl", "grpo-groups-index-v3"),
  ]);
  const runs = trainedRuns(runsIndex);
  const query = new URLSearchParams(location.search);
  const open = runs.find((r) => r.run === query.get("run"));
  if (query.get("rollout")) {
    const entry = open && stepsIndex.find((e) =>
      e.run === open.run && e.ts === open.ts && e.split === "train" && String(e.step) === query.get("step"));
    hideLoading();
    if (!entry) {
      showEmpty(`No training step ${query.get("step") ?? ""} in ${query.get("run") ?? "(no run)"}.`);
    } else {
      // The rail lists the opened attempt's group only, so j / k move between its siblings.
      const rows = await fetchJsonl(entry.path);
      const opened = rows.find((r) => String(r.index) === query.get("rollout")) ?? rows[0];
      await openRollout({
        entry, rows: rows.filter((r) => r.group === opened.group), method: methodText(open),
        onClose: () => {
          const back = new URLSearchParams(location.search);
          back.delete("rollout");
          back.delete("token");
          location.href = `attempts.html?${back}`;
        },
      });
    }
  } else {
    const kept = new Set(runs.map((r) => `${r.run}/${r.ts}`));
    const rows = groupsIndex.filter((g) => kept.has(`${g.run}/${g.ts}`)).flatMap(attemptsOf);
    const table = createTable({
      head: document.querySelector("#table-head"),
      body: document.querySelector("#table-body"),
      columns: COLUMNS,
      // Newest first: the latest run's last step on top, its attempts in order.
      defaultSort: [{key: "ts", desc: true}, {key: "step", desc: true}, {key: "index"}],
      pageSize: 1000,
      onOpen: (row) => {
        const next = new URLSearchParams(location.search);
        next.set("run", row.run);
        next.set("step", String(row.step));
        next.set("rollout", String(row.index));
        location.href = `attempts.html?${next}`;
      },
      onRender: (shown) => summarize([
        [`${formatNumber(shown.length)} attempts`, ""],
        [`${formatNumber(percent(shown.filter((r) => r.reward > 0.5).length, shown.length), 1)}% correct`, ""],
        [`${formatNumber(shown.reduce((n, r) => n + r.output_tokens, 0))} generated`, "completion-count"],
      ]),
    });
    if (open) {
      addInfoField("run", open.run);
      addInfoField("method", methodText(open));
    }
    for (const field of ["step", "group"]) {
      if (table.filters.has(field)) addInfoField(field, table.filters.get(field));
    }
    hideLoading();
    if (!rows.length) showEmpty();
    table.setRows(rows);
  }
} catch (error) {
  fatal(error);
}
