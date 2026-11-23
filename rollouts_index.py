"""
Summarize every saved rollout step into rollouts-index.jsonl, and every run into runs-index.jsonl (stdlib only).

    python rollouts_index.py                    # build once
    python rollouts_index.py --watch            # ./serve.sh runs this beside the server

rollouts-index.jsonl: a {"type": "meta"} line, then one line per output/rollouts/<run>/<timestamp>/<split>/step_NNNN.jsonl.
runs-index.jsonl: a {"type": "meta"} line, then one line per output/tb/<run>/<timestamp>/ (a training run) and one per
output/zero_shot/L<len>.json (a zero-shot baseline).
groups-index.jsonl: a {"type": "meta"} line, then one line per group of a training step: the prompt's group_size
rollouts, summarized so the groups tab need not fetch every step file.
Unchanged step files are not re-read: their entries (and groups) are reused when the path, size and mtime match.
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "grpo-rollouts-index-v2"
RUNS_SCHEMA = "grpo-runs-index-v1"
GROUPS_SCHEMA = "grpo-groups-index-v2"
ROOT = Path(__file__).parent
ROLLOUTS = ROOT / "output" / "rollouts"
INDEX = ROOT / "rollouts-index.jsonl"
RUNS_INDEX = ROOT / "runs-index.jsonl"
GROUPS_INDEX = ROOT / "groups-index.jsonl"


def read_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(len(rows), 1)
    rewards = [r["reward"] for r in rows]
    entry = {
        "count": len(rows),
        "correct": sum(x > 0.5 for x in rewards),  # a shaped reward scores a correct answer just under 1.0
        "partial": sum(abs(x - 0.1) < 1e-8 for x in rewards),
        "wrong": sum(abs(x) < 0.05 for x in rewards),
        "mean_reward": round(sum(rewards) / n, 4),
        "truncated": sum(r.get("finish_reason") == "length" for r in rows),
        "completion_tokens": sum(r.get("output_tokens", 0) for r in rows),
    }
    if rows and "group" in rows[0]:
        groups: Dict[int, set] = {}
        for r in rows:
            groups.setdefault(r["group"], set()).add(r["reward"])
        entry["groups"] = len(groups)
        entry["flat_groups"] = sum(len(v) == 1 for v in groups.values())  # zero advantage: no gradient from the group
        # A rollout whose advantage is zero adds nothing to the gradient, so only the rest count as trained on.
        entry["trained_tokens"] = sum(r.get("output_tokens", 0) for r in rows if r.get("advantage"))
    # Training minus rollout log-prob over the tokens both saw: how far the trainer's forward pass is from the engine.
    diffs = [t - s for r in rows if r.get("train_logprobs") and r.get("logprobs") and "train_tokens" not in r
             for t, s in zip(r["train_logprobs"], r["logprobs"])]
    if diffs:
        entry["logprob_diff"] = round(sum(diffs) / len(diffs), 6)
        entry["logprob_abs_diff"] = round(sum(abs(d) for d in diffs) / len(diffs), 6)
    return entry


def summarize_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        if "group" in r:
            groups.setdefault(r["group"], []).append(r)
    summaries = []
    for group, members in sorted(groups.items()):
        members.sort(key=lambda r: r.get("sample", 0))
        rewards = [r["reward"] for r in members]
        n = len(members)
        mean = sum(rewards) / n
        summaries.append({
            "group": group, "target": members[0]["target"], "numbers": members[0]["numbers"], "count": n,
            "correct": sum(x > 0.5 for x in rewards),
            "answered": sum(r.get("equation") is not None for r in members),
            "truncated": sum(r.get("finish_reason") == "length" for r in members),
            "mean_reward": round(mean, 4),
            "reward_std": round((sum((x - mean) ** 2 for x in rewards) / n) ** 0.5, 4),
            "flat": len(set(rewards)) == 1,  # zero advantage: no gradient from the group
            "completion_tokens": sum(r.get("output_tokens", 0) for r in members),
            # Each attempt in sample order, so the groups tab can draw one cell per attempt.
            "rewards": rewards,
            "truncations": [r.get("finish_reason") == "length" for r in members],
        })
    return summaries


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def meta(schema: str, total: int) -> Dict[str, Any]:
    return {"type": "meta", "schema": schema, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": total}


def read_previous_groups() -> Dict[str, List[Dict[str, Any]]]:
    previous: Dict[str, List[Dict[str, Any]]] = {}
    if not GROUPS_INDEX.exists():
        return previous
    for line in GROUPS_INDEX.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "meta" and row.get("schema") != GROUPS_SCHEMA:
            return {}
        if row.get("type") == "group":
            previous.setdefault(row.pop("path"), []).append(row)
    return previous


def refresh_rollouts() -> bool:
    previous: Dict[str, Dict[str, Any]] = {}
    previous_groups = read_previous_groups()
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "meta" and row.get("schema") != SCHEMA:
                break
            if row.get("type") == "entry":
                previous[row["path"]] = row
    entries: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(ROLLOUTS.glob("*/*/*/step_*.jsonl")):
        run, timestamp, split = path.parts[-4:-1]
        stat = path.stat()
        key = path.relative_to(ROOT).as_posix()
        old = previous.get(key)
        # A train step's groups are cached beside its entry; an index without them re-reads the file once.
        if old and old["size"] == stat.st_size and old["mtime"] == int(stat.st_mtime) and \
                (split != "train" or key in previous_groups):
            entries.append(old)
            if split == "train":
                groups[key] = previous_groups[key]
            continue
        rows = read_rows(path)
        entry = {"type": "entry", "path": key, "run": run, "ts": int(timestamp), "split": split,
                 "step": int(path.stem.split("_")[1]), "size": stat.st_size, "mtime": int(stat.st_mtime)}
        entries.append({**entry, **summarize(rows)})
        if split == "train":
            groups[key] = [{"type": "group", "run": run, "ts": int(timestamp), "step": entry["step"], **g}
                           for g in summarize_groups(rows)]
    if [e["path"] for e in entries] == list(previous) and all(e is previous[e["path"]] for e in entries) and \
            list(groups) == list(previous_groups):
        return False
    write_jsonl(INDEX, [meta(SCHEMA, len(entries)), *entries])
    group_rows = [{**g, "path": path} for path, rows in groups.items() for g in rows]
    write_jsonl(GROUPS_INDEX, [meta(GROUPS_SCHEMA, len(group_rows)), *group_rows])
    return True


def refresh_runs() -> bool:
    runs: List[Dict[str, Any]] = []
    for config in sorted((ROOT / "output" / "tb").glob("*/*/config.json")):
        folder = config.parent
        runs.append({"type": "run", "run": folder.parent.name, "ts": int(folder.name),
                     "base": folder.relative_to(ROOT).as_posix(), "config": json.loads(config.read_text())})
    for result in sorted((ROOT / "output" / "zero_shot").glob("L*.json")):
        runs.append({"type": "zero_shot", **json.loads(result.read_text())})
    rows = [meta(RUNS_SCHEMA, len(runs)), *runs]
    text = [json.dumps(row, separators=(",", ":")) for row in runs]
    if RUNS_INDEX.exists():
        old = RUNS_INDEX.read_text(encoding="utf-8").splitlines()[1:]
        if old == text:
            return False
    write_jsonl(RUNS_INDEX, rows)
    return True


def refresh() -> bool:
    return refresh_rollouts() | refresh_runs()


def server_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="keep refreshing until the server exits")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--server-pid", type=int, default=None)
    args = parser.parse_args()
    print(f"{'wrote' if refresh() else 'unchanged'} {INDEX.name}, {GROUPS_INDEX.name} and {RUNS_INDEX.name}", flush=True)
    while args.watch and server_alive(args.server_pid):
        time.sleep(args.interval)
        if refresh():
            print(f"{time.strftime('%H:%M:%S')} wrote {INDEX.name}, {GROUPS_INDEX.name} and {RUNS_INDEX.name}", flush=True)
