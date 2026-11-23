// What the five tabs (runs, metrics, steps, groups, attempts) share: loading the indexes, the URL-borne filters with *
// wildcards, the sortable table, the rail's fields and the tooltip. Like arc3-coach's viewers, no library.

export const tooltipEl = document.querySelector("#tooltip");
export const summaryEl = document.querySelector("#summary");
export const openEntryEl = document.querySelector("#open-entry");

// Every data path is relative to the repository root. ./serve.sh serves the pages at the root, but a static host such
// as GitHub Pages serves them from viewer/, so the root is one level up there.
const ROOT = new URL(location.pathname.includes("/viewer/") ? "../" : "./", location.href);

export async function fetchJsonl(path) {
  let response;
  try {
    response = await fetch(new URL(path, ROOT), {cache: "no-store"});
  } catch (error) {
    throw new Error(`${path}: ${error.message}`);
  }
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  // A step file is renamed into place whole, but a metrics file is appended to while its run trains, so a torn last
  // line is skipped rather than fatal.
  return (await response.text()).split(/\r?\n/).filter((line) => line.trim()).flatMap((line) => {
    try { return [JSON.parse(line)]; } catch { return []; }
  });
}

export async function loadIndex(path, schema) {
  const rows = await fetchJsonl(path);
  const meta = rows.find((row) => row.type === "meta");
  if (!meta || meta.schema !== schema) throw new Error(`${path}: expected schema ${schema}; run ./serve.sh`);
  return rows.filter((row) => row.type !== "meta");
}

// The run a person means by its name: the newest start of it, unless the URL pins a ts.
export function newestByRun(rows) {
  const newest = new Map();
  for (const row of rows) {
    if (!newest.has(row.run) || newest.get(row.run) < row.ts) newest.set(row.run, row.ts);
  }
  return newest;
}

export function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits});
}

// Signed, at `digits` places; one too small to show there (a length-shaped reward's) keeps its size in e-notation
// rather than reading as zero.
export function formatAdvantage(value, digits = 3) {
  const sign = value > 0 ? "+" : "";
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return `${sign}${value.toExponential(1)}`;
  return `${sign}${value.toFixed(digits)}`;
}

export function percent(part, whole) {
  return whole ? (100 * part) / whole : NaN;
}

export function addInfoField(label, value) {
  const field = document.createElement("div");
  field.className = "open-field";
  field.dataset.field = label;
  const name = document.createElement("div");
  name.className = "open-label";
  name.textContent = label;
  const content = document.createElement("div");
  if (value instanceof Node) content.append(value);
  else content.textContent = value;
  field.append(name, content);
  openEntryEl.append(field);
  return field;
}

export function moveTooltip(event) {
  const left = Math.min(event.clientX + 12, innerWidth - tooltipEl.offsetWidth - 10);
  const top = Math.min(event.clientY + 14, innerHeight - tooltipEl.offsetHeight - 10);
  tooltipEl.style.left = `${Math.max(8, left)}px`;
  tooltipEl.style.top = `${Math.max(8, top)}px`;
}

export function showTooltip(text, event) {
  tooltipEl.textContent = text;
  tooltipEl.style.display = "block";
  moveTooltip(event);
}

export function hideTooltip() {
  tooltipEl.style.display = "none";
}

// A filter rides in the query under the plain field name, so a filtered table survives the reload that picks up
// whatever the runs have written since. A value may hold * wildcards: run=dr_* keeps every Dr.GRPO run.
const patterns = new Map();

export function filterMatches(actual, wanted) {
  if (!wanted.includes("*")) return String(actual) === wanted;
  let pattern = patterns.get(wanted);
  if (!pattern) {
    const escaped = wanted.split("*").map((piece) => piece.replace(/[.+?^${}()|[\]\\]/g, "\\$&"));
    pattern = new RegExp(`^${escaped.join(".*")}$`);
    patterns.set(wanted, pattern);
  }
  return pattern.test(String(actual ?? ""));
}

export function readQuery(fields) {
  const params = new URLSearchParams(location.search);
  const values = new Map();
  for (const field of fields) {
    const value = params.get(field);
    if (value) values.set(field, value);
  }
  return values;
}

export function writeQuery(values, fields) {
  const params = new URLSearchParams(location.search);
  for (const field of fields) {
    const value = values.get(field);
    if (value !== undefined && value !== null && value !== "") params.set(field, value);
    else params.delete(field);
  }
  const query = params.toString();
  history.replaceState(null, "", location.pathname + (query ? `?${query}` : ""));
}

// A sortable, filterable table over index rows. A column is {key, label, num?, filter?, title?, text?(row),
// value?(row), render?(row)}: `value` is what sorts and filters, `text` what the cell reads, `render` a node drawn in
// its place. Filters and the sort live in the URL. With pageSize, only that many rows are drawn at first (tens of
// thousands of rows would stall the page), and a last row draws the next page; filters and sorts still see every row.
// rowClass(row) may name a class for a row.
export function createTable({head, body, columns, onOpen, defaultSort, onRender, pageSize = Infinity, rowClass}) {
  const filterFields = columns.filter((c) => c.filter).map((c) => c.key);
  const filters = readQuery(filterFields);
  const params = new URLSearchParams(location.search);
  let sort = params.get("sort") ? {key: params.get("sort").replace(/^-/, ""), desc: params.get("sort").startsWith("-")}
    : null;
  let rows = [];
  let limit = pageSize;

  // A default sort may name a field no column shows (a run's start time), which then sorts by its raw value.
  const columnOf = (key) => columns.find((c) => c.key === key) ?? {key};
  const valueOf = (column, row) => (column.value ? column.value(row) : row[column.key]);

  head.replaceChildren(...columns.map((column) => {
    const th = document.createElement("th");
    th.textContent = column.label;
    th.dataset.sort = column.key;
    if (column.num) th.className = "num";
    if (column.title) th.title = column.title;
    th.addEventListener("click", () => {
      sort = sort && sort.key === column.key ? {key: column.key, desc: !sort.desc} : {key: column.key, desc: !!column.num};
      const query = new Map([["sort", `${sort.desc ? "-" : ""}${sort.key}`]]);
      writeQuery(query, ["sort"]);
      limit = pageSize;
      render();
    });
    return th;
  }));

  function cell(column, row) {
    const td = document.createElement("td");
    if (column.num) td.className = "num";
    const value = valueOf(column, row);
    const text = column.text ? column.text(row) : (value ?? "—");
    if (column.render) {
      td.append(column.render(row));
    } else if (column.filter && value !== undefined && value !== null) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-value";
      button.textContent = text;
      const active = filters.get(column.key) === String(value);
      button.title = active ? `Show every ${column.label.toLowerCase()} again` : `Show only ${column.label.toLowerCase()} ${value}`;
      // The row beneath is the way in, so a filter click must not also open the row it landed on.
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        if (filters.get(column.key) === String(value)) filters.delete(column.key);
        else filters.set(column.key, String(value));
        writeQuery(filters, filterFields);
        limit = pageSize;
        render();
      });
      td.append(button);
    } else {
      td.textContent = text;
    }
    return td;
  }

  function render() {
    let shown = rows.filter((row) => [...filters].every(([key, wanted]) => {
      const column = columns.find((c) => c.key === key);
      return filterMatches(valueOf(column, row), wanted);
    }));
    const order = sort ? [sort] : defaultSort;
    shown = [...shown].sort((a, b) => {
      for (const {key, desc} of order) {
        const column = columnOf(key);
        const x = valueOf(column, a), y = valueOf(column, b);
        if (x === y) continue;
        if (x === undefined || x === null || Number.isNaN(x)) return 1;
        if (y === undefined || y === null || Number.isNaN(y)) return -1;
        return (x > y ? 1 : -1) * (desc ? -1 : 1);
      }
      return 0;
    });
    const drawn = shown.slice(0, limit).map((row) => {
      const tr = document.createElement("tr");
      const cls = rowClass?.(row);
      if (cls) tr.className = cls;
      tr.append(...columns.map((column) => cell(column, row)));
      tr.addEventListener("click", () => onOpen(row));
      return tr;
    });
    if (shown.length > limit) drawn.push(moreRow(shown.length));
    body.replaceChildren(...drawn);
    onRender?.(shown, filters);
  }

  function moreRow(total) {
    const tr = document.createElement("tr");
    tr.className = "more-row";
    const td = document.createElement("td");
    td.colSpan = columns.length;
    const next = Math.min(pageSize, total - limit);
    td.textContent = `Showing ${formatNumber(limit)} of ${formatNumber(total)} · show ${formatNumber(next)} more`;
    tr.append(td);
    tr.addEventListener("click", () => {
      limit += pageSize;
      render();
    });
    return tr;
  }

  return {
    setRows(next) { rows = next; render(); },
    render,
    filters,
  };
}

// A step file keeps token ids only; their text comes from the committed tokenizer, as in arc3-coach. One fetch
// decodes every rollout on the page.
const TOKENIZER_PATH = "tokenizer/tokenizer.json";
let vocab = null;

// Byte-level BPE writes each byte as a printable character; this is that mapping read backwards.
function byteAlphabet() {
  const codes = new Map();
  for (const [start, end] of [[33, 127], [161, 173], [174, 256]]) {
    for (let byte = start; byte < end; byte += 1) codes.set(byte, byte);
  }
  let spare = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!codes.has(byte)) codes.set(byte, 256 + spare++);
  }
  const alphabet = new Map();
  for (const [byte, code] of codes) alphabet.set(String.fromCharCode(code), byte);
  return alphabet;
}

function pieceBytes(piece, alphabet) {
  const bytes = new Uint8Array(piece.length);
  for (let index = 0; index < piece.length; index += 1) {
    const byte = alphabet.get(piece[index]);
    if (byte === undefined) return new TextEncoder().encode(piece);
    bytes[index] = byte;
  }
  return bytes;
}

export async function loadVocab() {
  if (vocab) return;
  let tokenizer;
  try {
    const response = await fetch(new URL(TOKENIZER_PATH, ROOT));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    tokenizer = await response.json();
  } catch (error) {
    throw new Error(`${TOKENIZER_PATH}: ${error.message}`);
  }
  if (tokenizer?.model?.type !== "BPE" || !tokenizer.model.vocab) {
    throw new Error(`${TOKENIZER_PATH} is not the expected BPE tokenizer`);
  }
  const alphabet = byteAlphabet();
  const next = [];
  for (const [piece, id] of Object.entries(tokenizer.model.vocab)) next[id] = pieceBytes(piece, alphabet);
  const encoder = new TextEncoder();
  for (const added of tokenizer.added_tokens || []) next[added.id] = encoder.encode(added.content);
  vocab = next;
}

// Streaming across pieces keeps a character whose UTF-8 bytes cross a token boundary whole, while still leaving one
// piece per token id.
export function decodeTokenIds(tokenIds) {
  if (!vocab) throw new Error("Tokenizer has not loaded");
  const decoder = new TextDecoder("utf-8", {fatal: false});
  const decoded = tokenIds.map((id) => {
    const bytes = vocab[id];
    if (bytes === undefined) throw new Error(`Tokenizer has no token id ${id}`);
    return decoder.decode(bytes, {stream: true});
  });
  const tail = decoder.decode();
  if (tail && decoded.length) decoded[decoded.length - 1] += tail;
  return decoded;
}

export const SHADE_STEPS = 32;

// Probability mass approaches one as a log-prob approaches zero. Its complement is a bounded darkness scale, so
// surprising tokens stand out without one outlier making the rest of the sequence indistinguishable.
export function shadeStep(logprob) {
  const darkness = 1 - Math.exp(Math.min(0, logprob));
  return Math.min(SHADE_STEPS - 1, Math.floor(darkness * SHADE_STEPS));
}

// A log-prob of zero is a token the model never hesitated over, so the first step is the page's own white.
export function installShades() {
  const style = document.createElement("style");
  style.textContent = Array.from({length: SHADE_STEPS}, (_unused, step) =>
    `.token.shade-${step} { --completion-token-background: hsl(48 96% ${(100 - 53 * step / (SHADE_STEPS - 1)).toFixed(2)}%); }`,
  ).join("\n");
  document.head.append(style);
}

export function fatal(error) {
  const empty = document.querySelector("#empty-state");
  document.querySelector("#loading-state").style.display = "none";
  empty.style.display = "block";
  empty.textContent = error.message || String(error);
}

// The rail's lines under the nav: what the table shows right now.
export function summarize(lines) {
  summaryEl.replaceChildren();
  for (const [text, cls] of lines) {
    const line = document.createElement("div");
    line.textContent = text;
    if (cls) line.className = cls;
    summaryEl.append(line);
  }
}

export const METHODS = {grpo: "grpo", dr_grpo: "dr-grpo", grpo_no_len_norm: "grpo-no-len-norm"};

export function methodText(run) {
  return `${METHODS[run.config.loss_type] ?? run.config.loss_type} · L${run.config.max_tokens}`;
}

// A run too short to compare (a smoke test) is left out of every tab.
export const MIN_STEPS = 10;

// The runs the tabs show: each name's newest start, trained for at least MIN_STEPS.
export function trainedRuns(runsIndex) {
  const runs = runsIndex.filter((r) => r.type === "run");
  const newest = newestByRun(runs);
  return runs.filter((r) => newest.get(r.run) === r.ts && r.config.n_grpo_steps >= MIN_STEPS);
}

export function hideLoading() {
  document.querySelector("#loading-state").style.display = "none";
}

export function showEmpty(text) {
  const empty = document.querySelector("#empty-state");
  if (text) empty.textContent = text;
  empty.style.display = "block";
}
