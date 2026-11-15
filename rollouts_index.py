"""
Summarize every saved rollout step into rollouts-index.jsonl, and every run into runs-index.jsonl (stdlib only).

    python rollouts_index.py                    # build once
    python rollouts_index.py --watch            # ./serve.sh runs this beside the server

rollouts-index.jsonl: a {"type": "meta"} line, then one line per output/rollouts/<run>/<timestamp>/<split>/step_NNNN.jsonl.
runs-index.jsonl: a {"type": "meta"} line, then one line per output/tb/<run>/<timestamp>/ (a training run) and one per
output/zero_shot/L<len>.json (a zero-shot baseline).
Unchanged step files are not re-read: their entries are reused when the path, size and mtime match.
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "grpo-rollouts-index-v2"
RUNS_SCHEMA = "grpo-runs-index-v1"
ROOT = Path(__file__).parent
ROLLOUTS = ROOT / "output" / "rollouts"
INDEX = ROOT / "rollouts-index.jsonl"
RUNS_INDEX = ROOT / "runs-index.jsonl"


def summarize(path: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = max(len(rows), 1)
    rewards = [r["reward"] for r in rows]
    entry = {
        "count": len(rows),
        "correct": sum(abs(x - 1.0) < 1e-8 for x in rewards),
        "partial": sum(abs(x - 0.1) < 1e-8 for x in rewards),
        "wrong": sum(abs(x) < 1e-8 for x in rewards),
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
    # Training minus rollout log-prob over the tokens both saw: how far the trainer's forward pass is from the engine.
    diffs = [t - s for r in rows if r.get("train_logprobs") and r.get("logprobs") and "train_tokens" not in r
             for t, s in zip(r["train_logprobs"], r["logprobs"])]
    if diffs:
        entry["logprob_diff"] = round(sum(diffs) / len(diffs), 6)
        entry["logprob_abs_diff"] = round(sum(abs(d) for d in diffs) / len(diffs), 6)
    return entry


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def meta(schema: str, total: int) -> Dict[str, Any]:
    return {"type": "meta", "schema": schema, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": total}


def refresh_rollouts() -> bool:
    previous: Dict[str, Dict[str, Any]] = {}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "meta" and row.get("schema") != SCHEMA:
                break
            if row.get("type") == "entry":
                previous[row["path"]] = row
    entries: List[Dict[str, Any]] = []
    for path in sorted(ROLLOUTS.glob("*/*/*/step_*.jsonl")):
        run, timestamp, split = path.parts[-4:-1]
        stat = path.stat()
        key = path.relative_to(ROOT).as_posix()
        old = previous.get(key)
        if old and old["size"] == stat.st_size and old["mtime"] == int(stat.st_mtime):
            entries.append(old)
            continue
        entries.append({"type": "entry", "path": key, "run": run, "ts": int(timestamp), "split": split,
                        "step": int(path.stem.split("_")[1]), "size": stat.st_size, "mtime": int(stat.st_mtime),
                        **summarize(path)})
    if [e["path"] for e in entries] == list(previous) and all(e is previous[e["path"]] for e in entries):
        return False
    write_jsonl(INDEX, [meta(SCHEMA, len(entries)), *entries])
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
    print(f"{'wrote' if refresh() else 'unchanged'} {INDEX.name} and {RUNS_INDEX.name}", flush=True)
    while args.watch and server_alive(args.server_pid):
        time.sleep(args.interval)
        if refresh():
            print(f"{time.strftime('%H:%M:%S')} wrote {INDEX.name} and {RUNS_INDEX.name}", flush=True)
