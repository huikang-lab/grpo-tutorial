// GRPO rollout viewer. Three levels, all kept in the URL query:
//   (nothing)                       every saved step, from output/rollouts-index.jsonl
//   ?run=&ts=&kind=&step=           the rollouts of one step file
//   ?run=&ts=&kind=&step=&i=        the decoded tokens of one rollout, shaded by log-prob
// Clicking a cell marked as filterable toggles a filter on that value.

const INDEX_PATH = "output/rollouts-index.jsonl";
const INDEX_SCHEMA = "grpo-rollouts-index-v1";
const SHADE_STEPS = 32;

const $ = (id) => document.getElementById(id);
const state = { index: [], rows: null, rowsKey: null, sort: {}, filters: {} };

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function shadeStep(logprob) {
  const darkness = 1 - Math.exp(Math.min(0, logprob));
  return Math.min(SHADE_STEPS - 1, Math.floor(darkness * SHADE_STEPS));
}

function shadeBackground(step) {
  const lightness = 100 - 53 * step / (SHADE_STEPS - 1);
  return `hsl(48 96% ${lightness.toFixed(2)}%)`;
}

const shadeStyle = document.createElement("style");
shadeStyle.textContent = Array.from({ length: SHADE_STEPS }, (_, s) =>
  `.token.shade-${s} { --completion-token-background: ${shadeBackground(s)}; }`).join("\n");
document.head.append(shadeStyle);

function rewardBadge(reward) {
  const kind = Math.abs(reward - 1) < 1e-8 ? "correct" : Math.abs(reward - 0.1) < 1e-8 ? "partial" : "failed";
  return `<span class="reward reward-${kind}">${reward.toFixed(1)}</span>`;
}

function advantage(value) {
  if (value === undefined) return '<span class="muted">—</span>';
  const cls = value > 0 ? "adv-pos" : value < 0 ? "adv-neg" : "muted";
  return `<span class="${cls}">${value > 0 ? "+" : ""}${value.toFixed(3)}</span>`;
}

function params() {
  return Object.fromEntries(new URLSearchParams(location.search));
}

function go(query) {
  const search = new URLSearchParams(Object.entries(query).filter(([, v]) => v !== undefined && v !== null));
  history.pushState(null, "", search.toString() ? `?${search}` : location.pathname);
  state.filters = {};
  render();
}

async function fetchJsonl(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.text()).split("\n").filter((line) => line.trim()).flatMap((line) => {
    try { return [JSON.parse(line)]; } catch { return []; }  // a half-written last line
  });
}

async function loadIndex() {
  const rows = await fetchJsonl(INDEX_PATH);
  const meta = rows.find((r) => r.type === "meta");
  if (!meta || meta.schema !== INDEX_SCHEMA) throw new Error(`expected index schema ${INDEX_SCHEMA}`);
  state.index = rows.filter((r) => r.type === "entry");
}

function stepPath(p) {
  return `output/rollouts/${p.run}/${p.ts}/${p.kind}/step_${String(p.step).padStart(4, "0")}.jsonl`;
}

async function loadRows(p) {
  const key = stepPath(p);
  if (state.rowsKey !== key) {
    state.rows = await fetchJsonl(key);
    state.rowsKey = key;
  }
  return state.rows;
}

// A column is {key, label, num?, filter?, html?(row), value?(row)}.
function renderTable(columns, rows, sortKey, onOpen, groupKey) {
  const sort = state.sort[sortKey] || {};
  const active = Object.entries(state.filters);
  let shown = rows.filter((row) => active.every(([k, v]) => String(cellValue(columns, k, row)) === v));
  if (sort.key) {
    const sign = sort.dir === "desc" ? -1 : 1;
    shown = [...shown].sort((a, b) => {
      const x = cellValue(columns, sort.key, a), y = cellValue(columns, sort.key, b);
      return (x > y ? 1 : x < y ? -1 : 0) * sign;
    });
  }
  const head = columns.map((c) => {
    const sorted = sort.key === c.key ? ` sorted-${sort.dir}` : "";
    return `<th data-sort="${c.key}" class="${c.num ? "num" : ""}${sorted}">${c.label}</th>`;
  }).join("");
  const body = shown.map((row) => {
    const odd = groupKey && row[groupKey] % 2 ? "group-odd" : "";
    const cells = columns.map((c) => {
      const html = c.html ? c.html(row) : escapeHtml(cellValue(columns, c.key, row) ?? "—");
      const filter = c.filter ? ` filterable" data-filter="${c.key}" data-value="${escapeHtml(cellValue(columns, c.key, row))}` : "";
      return `<td class="${c.num ? "num" : ""}${c.wrap ? " wrap" : ""}${filter}">${html}</td>`;
    }).join("");
    return `<tr class="${odd}" data-row="${rows.indexOf(row)}">${cells}</tr>`;
  }).join("");
  const wrap = $("table-wrap");
  wrap.innerHTML = shown.length ? `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`
    : `<div class="empty">No rows${active.length ? " match the filters" : ""}.</div>`;
  wrap.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
    const key = th.dataset.sort;
    const dir = sort.key === key && sort.dir === "asc" ? "desc" : "asc";
    state.sort[sortKey] = { key, dir };
    render();
  }));
  wrap.querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", (event) => {
    const cell = event.target.closest("[data-filter]");
    if (cell) {
      const { filter, value } = cell.dataset;
      if (state.filters[filter] === value) delete state.filters[filter];
      else state.filters[filter] = value;
      render();
      return;
    }
    onOpen(rows[Number(tr.dataset.row)]);
  }));
  renderFilters(columns);
  return shown;
}

function cellValue(columns, key, row) {
  const column = columns.find((c) => c.key === key);
  return column && column.value ? column.value(row) : row[key];
}

function renderFilters(columns) {
  $("filters").innerHTML = Object.entries(state.filters).map(([k, v]) => {
    const label = columns.find((c) => c.key === k)?.label ?? k;
    return `<button class="chip" data-key="${k}">${escapeHtml(label)}: ${escapeHtml(v)} ✕</button>`;
  }).join("");
  $("filters").querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    delete state.filters[chip.dataset.key];
    render();
  }));
}

function field(label, html) {
  return `<div class="open-field"><span class="open-label">${label}</span>${html}</div>`;
}

function renderNav(p) {
  const crumbs = ['<strong><a href="?">GRPO rollouts</a></strong>'];
  if (p.run) {
    crumbs.push(`<a href="?${new URLSearchParams({ run: p.run, ts: p.ts, kind: p.kind, step: p.step })}">` +
      `${escapeHtml(p.kind)} step ${escapeHtml(p.step)}</a>`);
  }
  if (p.i !== undefined) crumbs.push(`rollout ${escapeHtml(p.i)}`);
  $("nav").innerHTML = crumbs.join("<br>");
}

const STEP_COLUMNS = [
  { key: "run", label: "Run", filter: true },
  { key: "timestamp", label: "Started", filter: true, value: (r) => new Date(r.timestamp * 1000).toISOString().slice(5, 16).replace("T", " ") },
  { key: "kind", label: "Kind", filter: true },
  { key: "step", label: "Step", num: true },
  { key: "count", label: "Rollouts", num: true },
  { key: "accuracy", label: "Accuracy %", num: true, html: (r) => r.accuracy.toFixed(1) },
  { key: "mean_reward", label: "Mean reward", num: true, html: (r) => r.mean_reward.toFixed(3) },
  { key: "partial", label: "0.1", num: true },
  { key: "failed", label: "0.0", num: true },
  { key: "truncated", label: "Truncated", num: true },
  { key: "avg_tokens", label: "Avg tokens", num: true },
  { key: "flat_groups", label: "Flat groups", num: true, html: (r) => r.groups ? `${r.flat_groups}/${r.groups}` : '<span class="muted">—</span>' },
];

const ROLLOUT_COLUMNS = [
  { key: "index", label: "#", num: true },
  { key: "group", label: "Group", num: true, filter: true },
  { key: "sample", label: "Sample", num: true },
  { key: "target", label: "Target", num: true },
  { key: "numbers", label: "Numbers", value: (r) => r.numbers.join(", ") },
  { key: "equation", label: "Answer", wrap: true, value: (r) => r.equation ?? "" },
  { key: "result", label: "= ", num: true, value: (r) => (r.result === null ? "" : +r.result.toFixed(4)) },
  { key: "reward", label: "Reward", num: true, filter: true, html: (r) => rewardBadge(r.reward) },
  { key: "advantage", label: "Advantage", num: true, html: (r) => advantage(r.advantage) },
  { key: "output_tokens", label: "Tokens", num: true },
  { key: "finish_reason", label: "Finish", filter: true },
];

function renderSteps() {
  $("tokens").innerHTML = "";
  $("open-entry").innerHTML = "";
  const runs = new Set(state.index.map((e) => `${e.run}/${e.timestamp}`));
  $("summary").innerHTML = `<b>${runs.size}</b> runs<br><b>${state.index.length}</b> steps<br>` +
    `<b>${state.index.reduce((n, e) => n + e.count, 0).toLocaleString()}</b> rollouts`;
  if (!state.index.length) {
    $("table-wrap").innerHTML = '<div class="empty">No rollouts yet. Train with <code>python train_logged.py</code>.</div>';
    return;
  }
  if (!state.sort.steps) state.sort.steps = { key: "timestamp", dir: "desc" };
  renderTable(STEP_COLUMNS, state.index, "steps",
    (e) => go({ run: e.run, ts: e.timestamp, kind: e.kind, step: e.step }));
}

function renderRollouts(p, rows) {
  $("tokens").innerHTML = "";
  const entry = state.index.find((e) => e.run === p.run && String(e.timestamp) === p.ts && e.kind === p.kind && String(e.step) === p.step);
  $("summary").innerHTML = `<b>${rows.length}</b> rollouts`;
  $("open-entry").innerHTML = field("Run", escapeHtml(p.run)) + field("Kind", escapeHtml(p.kind)) +
    field("Step", escapeHtml(p.step)) + (entry ? field("Accuracy", `${entry.accuracy.toFixed(1)}%`) +
    field("Mean reward", entry.mean_reward.toFixed(3)) + field("Truncated", `${entry.truncated} of ${entry.count}`) : "");
  const columns = p.kind === "train" ? ROLLOUT_COLUMNS
    : ROLLOUT_COLUMNS.filter((c) => !["group", "sample", "advantage"].includes(c.key));
  renderTable(columns, rows, `rollouts-${p.kind}`, (r) => go({ ...p, i: r.index }), p.kind === "train" ? "group" : null);
}

function renderRollout(p, rows) {
  $("table-wrap").innerHTML = "";
  $("filters").innerHTML = "";
  const position = rows.findIndex((r) => String(r.index) === p.i);
  const r = rows[position];
  if (!r) {
    $("tokens").innerHTML = `<div class="empty">Rollout ${escapeHtml(p.i)} not found.</div>`;
    return;
  }
  const meanLogprob = r.logprobs && r.logprobs.length ? r.logprobs.reduce((a, b) => a + b, 0) / r.logprobs.length : null;
  $("summary").innerHTML = `<b>${position + 1}</b> of ${rows.length}<br><kbd>j</kbd> next · <kbd>k</kbd> previous`;
  $("open-entry").innerHTML = field("Reward", rewardBadge(r.reward)) +
    (r.advantage !== undefined ? field("Advantage", advantage(r.advantage)) : "") +
    (r.group !== undefined ? field("Group · sample", `${r.group} · ${r.sample}`) : "") +
    field("Target", r.target) + field("Numbers", escapeHtml(r.numbers.join(", "))) +
    field("Answer", r.equation === null ? '<span class="muted">no &lt;answer&gt;</span>' : `<code>${escapeHtml(r.equation)}</code>`) +
    field("Evaluates to", r.result === null ? '<span class="muted">—</span>' : +r.result.toFixed(6)) +
    field("Tokens", `${r.output_tokens} (${escapeHtml(r.finish_reason)})`) +
    (meanLogprob !== null ? field("Mean log-prob", meanLogprob.toFixed(3)) : "");
  const tokens = r.pieces.map((piece, k) => {
    const logprob = r.logprobs ? r.logprobs[k] : 0;
    return `<span class="token shade-${shadeStep(logprob)}" data-k="${k}">${escapeHtml(piece)}</span>`;
  }).join("");
  $("tokens").innerHTML = `<details class="prompt-block"><summary>Prompt</summary><div class="prompt">${escapeHtml(r.prompt)}</div></details>` +
    `<span class="segment-label">Completion · shaded by surprise (1 − p)</span>${tokens}`;
  const tooltip = $("tooltip");
  $("tokens").querySelectorAll(".token").forEach((span) => {
    const k = Number(span.dataset.k);
    span.addEventListener("mouseenter", () => {
      tooltip.textContent = `#${k}  id ${r.tokens[k]}` + (r.logprobs ? `\nlogprob ${r.logprobs[k].toFixed(3)}  p ${Math.exp(r.logprobs[k]).toFixed(3)}` : "");
      tooltip.hidden = false;
    });
    span.addEventListener("mousemove", (event) => {
      tooltip.style.left = `${event.clientX + 12}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
    });
    span.addEventListener("mouseleave", () => { tooltip.hidden = true; });
    span.addEventListener("click", () => span.classList.toggle("selected"));
  });
}

async function render() {
  const p = params();
  renderNav(p);
  try {
    if (!p.run) {
      await loadIndex();
      renderSteps();
    } else {
      if (!state.index.length) await loadIndex();
      const rows = await loadRows(p);
      if (p.i === undefined) renderRollouts(p, rows);
      else renderRollout(p, rows);
    }
  } catch (error) {
    $("table-wrap").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  }
}

document.addEventListener("keydown", (event) => {
  const p = params();
  if (p.i === undefined || !state.rows || event.target.closest?.("input, textarea")) return;
  const position = state.rows.findIndex((r) => String(r.index) === p.i);
  const next = event.key === "j" ? position + 1 : event.key === "k" ? position - 1 : null;
  if (next !== null && state.rows[next]) go({ ...p, i: state.rows[next].index });
  if (event.key === "Escape") go({ run: p.run, ts: p.ts, kind: p.kind, step: p.step });
});

window.addEventListener("popstate", render);
render();
