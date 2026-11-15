// What training.html and metrics.html share: loading the indexes, the URL-borne filters with *
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
// value?(row)}: `value` is what sorts and filters, `text` what the cell reads. Filters and the sort live in the URL.
export function createTable({head, body, columns, onOpen, defaultSort, onRender}) {
  const filterFields = columns.filter((c) => c.filter).map((c) => c.key);
  const filters = readQuery(filterFields);
  const params = new URLSearchParams(location.search);
  let sort = params.get("sort") ? {key: params.get("sort").replace(/^-/, ""), desc: params.get("sort").startsWith("-")}
    : null;
  let rows = [];

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
      render();
    });
    return th;
  }));

  function cell(column, row) {
    const td = document.createElement("td");
    if (column.num) td.className = "num";
    const value = valueOf(column, row);
    const text = column.text ? column.text(row) : (value ?? "—");
    if (column.filter && value !== undefined && value !== null) {
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
        const column = columns.find((c) => c.key === key);
        const x = valueOf(column, a), y = valueOf(column, b);
        if (x === y) continue;
        if (x === undefined || x === null || Number.isNaN(x)) return 1;
        if (y === undefined || y === null || Number.isNaN(y)) return -1;
        return (x > y ? 1 : -1) * (desc ? -1 : 1);
      }
      return 0;
    });
    body.replaceChildren(...shown.map((row) => {
      const tr = document.createElement("tr");
      tr.append(...columns.map((column) => cell(column, row)));
      tr.addEventListener("click", () => onOpen(row));
      return tr;
    }));
    onRender?.(shown, filters);
  }

  return {
    setRows(next) { rows = next; render(); },
    render,
    filters,
  };
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
