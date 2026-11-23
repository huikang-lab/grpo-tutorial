// One rollout of a training step: the rollouts it is given (its group's) down the rail, the selected one's prompt and completion as one
// token stream shaded by surprise (or by training minus rollout log-prob), and what it scored beside it with its group.
// The attempts tab opens it at ?run=&step=&rollout=; j / k step through the rollouts, Esc goes back to the step's table.

import {
  addInfoField, decodeTokenIds, formatAdvantage, formatNumber, hideTooltip, installShades, loadVocab, openEntryEl, percent, shadeStep,
  showTooltip, summaryEl,
} from "./common.js";

const tokenEl = document.querySelector("#token-content");
const sideEl = document.querySelector("#side-panel");
const state = {entry: null, rows: [], rollout: null, selectedToken: null, method: ""};

function params() {
  return new URLSearchParams(location.search);
}

// A token that is only a line break paints no glyph, so a return mark before each break gives it a box to hover;
// the mark is unselectable, so copied text keeps only the real characters.
function setTokenText(token, text) {
  const parts = text.split("\n");
  token.textContent = parts[0];
  for (const part of parts.slice(1)) {
    const mark = document.createElement("span");
    mark.className = "newline-mark";
    mark.textContent = "⏎";
    mark.setAttribute("aria-hidden", "true");
    token.append(mark, `\n${part}`);
  }
}

function aligned(row) {
  return Array.isArray(row.train_logprobs) && !row.train_tokens && row.logprobs;
}

// A group's eight rollouts share one prompt, which the step file keeps on sample 0 alone. Older runs kept it as text.
function promptOf(row) {
  const source = row.prompt_tokens || row.prompt ? row
    : state.rows.find((r) => r.group === row.group && (r.prompt_tokens || r.prompt));
  return {tokens: source?.prompt_tokens ?? null, text: source?.prompt ?? ""};
}

// Pieces are decoded once per row and kept on it, so hovering does not decode again.
function piecesOf(row) {
  row.decoded ??= {
    prompt: promptOf(row).tokens ? decodeTokenIds(promptOf(row).tokens) : null,
    completion: decodeTokenIds(row.tokens),
  };
  return row.decoded;
}

function tokenTooltip(row, kind, k) {
  const pieces = piecesOf(row);
  if (kind === "prompt") {
    return `prompt · pos ${k} · token ${promptOf(row).tokens[k].toLocaleString()} · ${JSON.stringify(pieces.prompt[k])}`;
  }
  const parts = [`completion · pos ${k} · token ${row.tokens[k].toLocaleString()} · ${JSON.stringify(pieces.completion[k])}`];
  const sampled = row.logprobs ? row.logprobs[k] : null;
  parts.push(Number.isFinite(sampled) ? `rollout logprob ${sampled.toFixed(6)} · p ${Math.exp(sampled).toFixed(4)}`
    : "rollout logprob unavailable");
  if (aligned(row)) {
    const trained = row.train_logprobs[k];
    parts.push(`training logprob ${trained.toFixed(6)} · diff ${(trained - sampled).toFixed(6)}`);
  } else if (row.train_tokens) {
    parts.push("training logprob: the trainer re-tokenized this response differently");
  }
  return parts.join("\n");
}

function shadeFor(row, k) {
  if (params().get("shade") === "diff" && aligned(row)) {
    // Shade by how far apart the two passes are, on the same white-to-amber scale: |diff| of 0 is white.
    return shadeStep(-Math.abs(row.train_logprobs[k] - row.logprobs[k]) * 10);
  }
  const logprob = row.logprobs ? row.logprobs[k] : null;
  return Number.isFinite(logprob) ? shadeStep(logprob) : null;
}

function selectToken(position) {
  state.selectedToken = position;
  tokenEl.querySelector(".token.selected")?.classList.remove("selected");
  const node = position === null ? null : tokenEl.querySelector(`.token[data-pos="${CSS.escape(position)}"]`);
  node?.classList.add("selected");
  const next = params();
  if (node) next.set("token", position);
  else next.delete("token");
  history.replaceState(null, "", `${location.pathname}?${next}`);
  return node;
}

function renderTokens(row) {
  const prompt = document.createElement("span");
  prompt.className = "segment prompt";
  const pieces = piecesOf(row);
  if (pieces.prompt) {
    pieces.prompt.forEach((piece, k) => prompt.append(tokenNode(row, "prompt", k, piece, null)));
  } else {
    prompt.textContent = promptOf(row).text;
  }
  const completion = document.createElement("span");
  completion.className = "segment completion";
  pieces.completion.forEach((piece, k) => completion.append(tokenNode(row, "completion", k, piece, shadeFor(row, k))));
  tokenEl.replaceChildren(prompt, completion);
  const wanted = params().get("token");
  if (wanted) selectToken(wanted)?.scrollIntoView({block: "center"});
}

function tokenNode(row, kind, k, piece, shade) {
  const token = document.createElement("span");
  token.className = shade === null ? "token" : `token completion-token shade-${shade}`;
  token.dataset.pos = `${kind === "prompt" ? "p" : "c"}${k}`;
  setTokenText(token, piece);
  token.addEventListener("mouseenter", (event) => showTooltip(tokenTooltip(row, kind, k), event));
  token.addEventListener("mousemove", (event) => showTooltip(tokenTooltip(row, kind, k), event));
  token.addEventListener("mouseleave", hideTooltip);
  token.addEventListener("click", () => selectToken(state.selectedToken === token.dataset.pos ? null : token.dataset.pos));
  return token;
}

function rewardText(reward) {
  return Number.isInteger(Math.round(reward * 1e6) / 1e5) ? reward.toFixed(1) : reward.toFixed(6);
}

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
}

function sideField(label, value) {
  const field = document.createElement("div");
  field.className = "open-field";
  const name = document.createElement("div");
  name.className = "open-label";
  name.textContent = label;
  const content = document.createElement("div");
  if (value instanceof Node) content.append(value);
  else content.textContent = value;
  field.append(name, content);
  sideEl.append(field);
}

function rolloutButton(row, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "call-link";
  button.dataset.rollout = String(row.index);
  button.textContent = label;
  if (state.rollout && state.rollout.index === row.index) button.classList.add("is-open");
  button.addEventListener("click", () => {
    const next = params();
    next.set("rollout", String(row.index));
    next.delete("token");
    history.replaceState(null, "", `${location.pathname}?${next}`);
    renderRollout();
  });
  return button;
}

function renderSide(row) {
  sideEl.replaceChildren();
  const heading = document.createElement("div");
  heading.className = "side-heading";
  heading.textContent = `Rollout ${String(row.index).padStart(4, "0")}`;
  sideEl.append(heading);
  sideField("reward", rewardText(row.reward));
  if (row.advantage !== undefined) sideField("advantage", formatAdvantage(row.advantage, 4));
  sideField("target", String(row.target));
  sideField("numbers", row.numbers.join(", "));
  sideField("answer", row.equation === null ? "no <answer>" : row.equation);
  sideField("value", row.result === null ? "—" : String(+row.result.toFixed(6)));
  const counts = document.createElement("div");
  const promptCount = promptOf(row).tokens ? promptOf(row).tokens.length : null;
  counts.innerHTML = `${promptCount === null ? "" : `<span class="prompt-count">${promptCount} prompt</span> · `}` +
    `<span class="completion-count">${row.output_tokens} completion</span>`;
  sideField("tokens", counts);
  sideField("finish", row.finish_reason === "length" ? "truncated at max_tokens" : "stopped");
  if (row.logprobs) sideField("rollout logprob", `${mean(row.logprobs).toFixed(4)} mean`);
  if (aligned(row)) {
    const diffs = row.train_logprobs.map((t, k) => t - row.logprobs[k]);
    sideField("logprob diff", `${mean(diffs).toFixed(6)} mean · ${mean(diffs.map(Math.abs)).toFixed(6)} mean |diff|`);
    const shade = document.createElement("div");
    shade.className = "call-list";
    for (const [value, label] of [[null, "shade by surprise (1 − p)"], ["diff", "shade by |training − rollout|"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "call-link";
      button.textContent = label;
      if ((params().get("shade") || null) === value) button.classList.add("is-open");
      button.addEventListener("click", () => {
        const next = params();
        if (value) next.set("shade", value);
        else next.delete("shade");
        history.replaceState(null, "", `${location.pathname}?${next}`);
        renderRollout();
      });
      shade.append(button);
    }
    sideField("shade", shade);
  } else if (row.group !== undefined) {
    sideField("logprob diff", row.train_tokens ? "re-tokenized differently" : "not recorded in this run");
  }
  if (row.group !== undefined) {
    const group = document.createElement("div");
    group.className = "call-list";
    for (const sibling of state.rows.filter((r) => r.group === row.group)) {
      group.append(rolloutButton(sibling, `sample ${sibling.sample} · ${rewardText(sibling.reward)} · ` +
        formatAdvantage(sibling.advantage, 2)));
    }
    sideField(`group ${row.group}`, group);
  }
}

function renderRail() {
  const entry = state.entry;
  summaryEl.replaceChildren();
  // Counted over the rollouts on the rail (a shaped reward scores a correct answer just under 1.0).
  for (const text of [`${formatNumber(state.rows.length)} rollouts`,
    `${formatNumber(percent(state.rows.filter((r) => r.reward > 0.5).length, state.rows.length), 1)}% correct`]) {
    const line = document.createElement("div");
    line.textContent = text;
    summaryEl.append(line);
  }
  openEntryEl.replaceChildren();
  addInfoField("run", entry.run);
  addInfoField("method", state.method);
  addInfoField("step", String(entry.step));
  const groups = new Set(state.rows.map((r) => r.group));
  if (groups.size === 1) addInfoField("group", String(state.rows[0].group));
  const field = addInfoField("rollouts", "");
  field.lastElementChild.remove();
  const list = document.createElement("div");
  list.className = "call-list";
  for (const row of state.rows) {
    list.append(rolloutButton(row, `${String(row.index).padStart(4, "0")} · ${rewardText(row.reward)}`));
  }
  field.append(list);
}

function renderRollout() {
  const wanted = params().get("rollout");
  state.rollout = state.rows.find((r) => String(r.index) === wanted) || state.rows[0];
  state.selectedToken = null;
  openEntryEl.querySelectorAll(".call-link").forEach((button) =>
    button.classList.toggle("is-open", button.dataset.rollout === String(state.rollout.index)));
  openEntryEl.querySelector(".call-link.is-open")?.scrollIntoView({block: "nearest"});
  renderTokens(state.rollout);
  renderSide(state.rollout);
}

export async function openRollout({entry, rows, method, onClose}) {
  await loadVocab();
  installShades();
  state.entry = entry;
  state.rows = rows;
  state.method = method;
  document.body.classList.add("has-detail");
  renderRail();
  renderRollout();
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey) return;
    if (event.key === "Escape") {
      onClose();
      return;
    }
    const step = {j: 1, k: -1}[event.key];
    if (!step || !state.rollout) return;
    const position = state.rows.indexOf(state.rollout) + step;
    if (position < 0 || position >= state.rows.length) return;
    const next = params();
    next.set("rollout", String(state.rows[position].index));
    next.delete("token");
    history.replaceState(null, "", `${location.pathname}?${next}`);
    renderRollout();
  });
}
