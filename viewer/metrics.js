// The metrics tab: every measure against the GRPO step, drawn as inline SVG. With no run named, every run on one set
// of charts with the zero-shot baselines as dashed lines; with ?run=, that run alone and its config down the rail.

import {
  addInfoField, fatal, fetchJsonl, formatNumber, hideTooltip, loadIndex, newestByRun, openEntryEl, showTooltip,
  summaryEl,
} from "./common.js";

const SVG_NS = "http://www.w3.org/2000/svg";
// Categorical slots in fixed order, validated for colour-vision separation on white. Two sit under 3:1 contrast, so
// every line also carries a direct label and a legend entry.
const RUN_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const INK = "#0f172a";
const REFERENCE = "#87919c";

// A chart is one measure against the step. `series(run)` returns its lines for a run: [{label, points}], where a
// label of null means the measure itself.
const CHARTS = [
  {title: "Eval accuracy", subtitle: "% of the 1024 test problems solved, one sample each", unit: "%",
   series: (run) => [{label: null, points: run.eval.map((r) => ({x: r.step, y: r.accuracy}))}], references: true},
  {title: "Reward", subtitle: "mean over the step's 128 training rollouts",
   series: (run) => [{label: null, points: run.train.map((r) => ({x: r.step, y: r.reward_mean}))}]},
  {title: "Completion length", subtitle: "mean tokens generated per training rollout",
   series: (run) => [{label: null, points: run.train.map((r) => ({x: r.step, y: r.avg_output_tokens}))}]},
  {title: "Truncated", subtitle: "% of training rollouts stopped at max_tokens", unit: "%",
   series: (run) => [{label: null, points: run.index.train.map((e) => ({x: e.step, y: (100 * e.truncated) / e.count}))}]},
  {title: "Zero-advantage groups", subtitle: "of 16: every sample scored the same, so the group teaches nothing",
   series: (run) => [{label: null, points: run.index.train.filter((e) => e.groups).map((e) => ({x: e.step, y: e.flat_groups}))}]},
  {title: "Logprob diff", subtitle: "training minus rollout log-prob per token: trainer vs engine",
   series: (run) => [{label: null, points: run.index.train.filter((e) => Number.isFinite(e.logprob_diff))
     .map((e) => ({x: e.step, y: e.logprob_diff}))}]},
  {title: "Loss", subtitle: "summed over the step's microbatches",
   series: (run) => [{label: null, points: run.train.map((r) => ({x: r.step, y: r.loss}))}]},
  {title: "Grad norm", subtitle: "L2, before clipping to 1.0",
   series: (run) => [{label: null, points: run.train.map((r) => ({x: r.step, y: r.grad_norm}))}]},
  {title: "Step time", subtitle: "seconds per step: generating, then training",
   series: (run) => (state.open
     ? [{label: "generate", points: run.train.map((r) => ({x: r.step, y: r.t_gen}))},
        {label: "train", points: run.train.map((r) => ({x: r.step, y: r.t_train}))}]
     : [{label: null, points: run.train.map((r) => ({x: r.step, y: r.t_gen + r.t_train}))}])},
];

const chartsEl = document.querySelector("#charts");
const loadingEl = document.querySelector("#loading-state");
const emptyEl = document.querySelector("#empty-state");
const state = {runs: [], zeroShot: [], open: null, hidden: new Set()};

function runLabel(run) {
  return run.run;
}

function svgElement(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, String(value));
  return node;
}

function formatValue(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 0.001 || magnitude >= 100_000)) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

// The lines a chart draws: one per run when every run is shown, the run's own measures when one is open.
function linesFor(chart) {
  const runs = state.open ? [state.open] : state.runs;
  const lines = [];
  for (const run of runs) {
    for (const series of chart.series(run)) {
      if (!series.points.length) continue;
      const label = state.open ? (series.label ?? chart.title) : `${runLabel(run)}${series.label ? ` ${series.label}` : ""}`;
      const colour = state.open
        ? (series.label === "train" ? RUN_COLOURS[1] : series.label ? RUN_COLOURS[0] : INK)
        : RUN_COLOURS[run.slot % RUN_COLOURS.length];
      // Every run's lines hide together, so the legend toggles a run rather than one of its measures.
      lines.push({key: state.open ? label : run.run, label, colour, points: series.points});
    }
  }
  return lines;
}

function referencesFor(chart) {
  if (!chart.references) return [];
  const lengths = state.open ? [state.open.config.max_tokens, 1024] : null;
  return state.zeroShot
    .filter((z) => !lengths || lengths.includes(z.max_tokens))
    .map((z) => ({label: `zero-shot L${z.max_tokens}`, value: z.accuracy}));
}

const PLOT = {left: 46, right: 14, top: 10, bottom: 20, width: 420, height: 150};

function drawChart(card, chart, lines, references) {
  const visible = lines.filter((line) => !state.hidden.has(line.key));
  const svg = svgElement("svg", {
    class: "chart-svg", viewBox: `0 0 ${PLOT.width} ${PLOT.height}`, preserveAspectRatio: "xMidYMid meet",
    role: "img", "aria-label": chart.title,
  });
  const xs = visible.flatMap((line) => line.points.map((p) => p.x));
  const ys = [...visible.flatMap((line) => line.points.map((p) => p.y)), ...references.map((r) => r.value)];
  if (!xs.length) {
    card.append(svg);
    return;
  }
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= Math.abs(yMin) * 0.1 || 0.1;
    yMax += Math.abs(yMax) * 0.1 || 0.1;
  }
  const plotWidth = PLOT.width - PLOT.left - PLOT.right;
  const plotHeight = PLOT.height - PLOT.top - PLOT.bottom;
  const xAt = (v) => PLOT.left + (xMax === xMin ? plotWidth / 2 : ((v - xMin) / (xMax - xMin)) * plotWidth);
  const yAt = (v) => PLOT.top + plotHeight - ((v - yMin) / (yMax - yMin)) * plotHeight;

  for (let tick = 0; tick <= 3; tick += 1) {
    const value = yMin + ((yMax - yMin) * tick) / 3;
    svg.append(svgElement("line", {class: "chart-grid", x1: PLOT.left, x2: PLOT.width - PLOT.right, y1: yAt(value), y2: yAt(value)}));
    const label = svgElement("text", {class: "chart-axis-text", x: PLOT.left - 6, y: yAt(value) + 3, "text-anchor": "end"});
    label.textContent = formatValue(value);
    svg.append(label);
  }
  for (const value of xMax === xMin ? [xMin] : [xMin, xMax]) {
    const label = svgElement("text", {
      class: "chart-axis-text", x: xAt(value), y: PLOT.height - 6, "text-anchor": value === xMin ? "start" : "end",
    });
    label.textContent = `step ${value}`;
    svg.append(label);
  }
  // The baselines a trained run is measured against: grey, dashed, and named at their left end.
  for (const reference of references) {
    const y = yAt(reference.value);
    svg.append(svgElement("line", {class: "chart-reference", stroke: REFERENCE, x1: PLOT.left, x2: PLOT.width - PLOT.right, y1: y, y2: y}));
    const label = svgElement("text", {class: "chart-label", fill: REFERENCE, x: PLOT.left + 4, y: y - 4});
    label.textContent = reference.label;
    svg.append(label);
  }
  for (const line of visible) {
    svg.append(svgElement("polyline", {
      class: "chart-line", stroke: line.colour, points: line.points.map((p) => `${xAt(p.x)},${yAt(p.y)}`).join(" "),
    }));
    if (line.points.length === 1) {
      svg.append(svgElement("circle", {class: "chart-point", cx: xAt(line.points[0].x), cy: yAt(line.points[0].y), r: 4, fill: line.colour}));
    }
    if (visible.length > 1 && line.points.length > 1) {
      const last = line.points.at(-1);
      const label = svgElement("text", {
        class: "chart-label", fill: line.colour, x: xAt(last.x) - 4, "text-anchor": "end",
        y: Math.min(PLOT.top + plotHeight - 2, Math.max(PLOT.top + 9, yAt(last.y) - 6)),
      });
      label.textContent = line.label;
      svg.append(label);
    }
  }
  const crosshair = svgElement("line", {class: "chart-crosshair", y1: PLOT.top, y2: PLOT.top + plotHeight, x1: PLOT.left, x2: PLOT.left, opacity: 0});
  svg.append(crosshair);
  svg.append(svgElement("rect", {class: "chart-hit", x: PLOT.left, y: PLOT.top, width: plotWidth, height: plotHeight}));
  const steps = [...new Set(xs)].sort((a, b) => a - b);
  svg.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const scale = box.width / PLOT.width;
    const value = xMin + ((event.clientX - box.left) / scale - PLOT.left) / plotWidth * (xMax - xMin);
    const step = steps.reduce((best, s) => (Math.abs(s - value) < Math.abs(best - value) ? s : best), steps[0]);
    crosshair.setAttribute("x1", String(xAt(step)));
    crosshair.setAttribute("x2", String(xAt(step)));
    crosshair.setAttribute("opacity", "1");
    const rows = visible.map((line) => {
      const point = line.points.find((p) => p.x === step);
      return point ? `${line.label} ${formatValue(point.y)}${chart.unit || ""}` : null;
    }).filter(Boolean);
    showTooltip([`step ${step}`, ...rows].join("\n"), event);
  });
  svg.addEventListener("mouseleave", () => {
    crosshair.setAttribute("opacity", "0");
    hideTooltip();
  });
  card.append(svg);
}

function chartCard(chart) {
  const lines = linesFor(chart);
  if (!lines.length) return null;
  const card = document.createElement("div");
  card.className = "chart-card";
  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = chart.title;
  const subtitle = document.createElement("div");
  subtitle.className = "chart-subtitle";
  subtitle.textContent = chart.subtitle;
  card.append(title, subtitle);
  drawChart(card, chart, lines, referencesFor(chart));
  // A legend whenever there is more than one line, so identity is never colour alone; clicking one takes its line
  // out of the scale.
  if (lines.length > 1) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const line of lines) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "legend-item";
      if (state.hidden.has(line.key)) item.classList.add("is-hidden");
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = line.colour;
      const label = document.createElement("span");
      label.textContent = line.label;
      item.append(swatch, label);
      item.addEventListener("click", () => {
        if (state.hidden.has(line.key)) state.hidden.delete(line.key);
        else state.hidden.add(line.key);
        renderCharts();
      });
      legend.append(item);
    }
    card.append(legend);
  }
  return card;
}

function renderCharts() {
  chartsEl.replaceChildren(...CHARTS.map(chartCard).filter(Boolean));
}

function runLink(run) {
  const link = document.createElement("a");
  link.className = "run-link";
  const params = new URLSearchParams();
  if (run) params.set("run", run.run);
  link.href = `${location.pathname}${run ? `?${params}` : ""}`;
  link.textContent = run ? run.run : "every run";
  if (run === state.open) link.classList.add("is-open");
  return link;
}

function configDetails(config) {
  const details = document.createElement("details");
  details.className = "config-dump";
  const summary = document.createElement("summary");
  summary.textContent = "all fields";
  const body = document.createElement("pre");
  body.textContent = JSON.stringify(config, null, 2);
  details.append(summary, body);
  return details;
}

function renderRail() {
  summaryEl.replaceChildren();
  const open = state.open;
  const lines = open
    ? [`${open.train.length} of ${open.config.n_grpo_steps} steps`,
       `final eval ${formatNumber(open.eval.at(-1)?.accuracy, 1)}%`,
       `best eval ${formatNumber(Math.max(...open.eval.map((r) => r.accuracy)), 1)}%`]
    : [`${state.runs.length} runs`, `${state.zeroShot.length} zero-shot evals`];
  for (const text of lines) {
    const line = document.createElement("div");
    line.textContent = text;
    summaryEl.append(line);
  }
  openEntryEl.replaceChildren();
  const field = addInfoField("run", "");
  field.lastElementChild.remove();
  const list = document.createElement("div");
  list.className = "run-list";
  list.append(runLink(null), ...state.runs.map(runLink));
  field.append(list);
  if (open) {
    addInfoField("loss", open.config.loss_type);
    addInfoField("max tokens", String(open.config.max_tokens));
    addInfoField("lr", `${open.config.lr} · seed ${open.config.seed}`);
    addInfoField("gpu", open.config.gpu ?? "—");
    addInfoField("config", configDetails(open.config));
  } else {
    for (const z of state.zeroShot) {
      addInfoField(`zero-shot L${z.max_tokens}`, `${formatNumber(z.accuracy, 2)}% · truncated ${formatNumber(100 * z.truncated_frac, 1)}%`);
    }
  }
}

async function loadRuns() {
  const [runsIndex, stepsIndex] = await Promise.all([
    loadIndex("runs-index.jsonl", "grpo-runs-index-v1"),
    loadIndex("rollouts-index.jsonl", "grpo-rollouts-index-v2"),
  ]);
  const runs = runsIndex.filter((r) => r.type === "run");
  const newest = newestByRun(runs);
  const latest = runs.filter((r) => newest.get(r.run) === r.ts).sort((a, b) => a.run.localeCompare(b.run));
  state.zeroShot = runsIndex.filter((r) => r.type === "zero_shot").sort((a, b) => a.max_tokens - b.max_tokens);
  state.runs = await Promise.all(latest.map(async (run, slot) => {
    const [train, evals] = await Promise.all([
      fetchJsonl(`${run.base}/train_metrics.jsonl`).catch(() => []),
      fetchJsonl(`${run.base}/eval_metrics.jsonl`).catch(() => []),
    ]);
    const mine = stepsIndex.filter((e) => e.run === run.run && e.ts === run.ts).sort((a, b) => a.step - b.step);
    return {...run, slot, train, eval: evals,
            index: {train: mine.filter((e) => e.split === "train"), eval: mine.filter((e) => e.split === "eval")}};
  }));
  // A run too short to read as a line (a smoke test) starts hidden; the legend brings it back.
  for (const run of state.runs) if (run.config.n_grpo_steps < 10) state.hidden.add(run.run);
}

try {
  await loadRuns();
  const wanted = new URLSearchParams(location.search).get("run");
  state.open = state.runs.find((run) => run.run === wanted) || null;
  if (state.open) state.hidden.clear();
  loadingEl.style.display = "none";
  if (!state.runs.length) emptyEl.style.display = "block";
  renderRail();
  renderCharts();
} catch (error) {
  fatal(error);
}
