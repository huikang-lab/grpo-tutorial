"""
Summarize every saved rollout step into output/rollouts-index.jsonl for the viewer (stdlib only).

    python rollouts_index.py            # serve.py also refreshes the index whenever the viewer loads it

The first line is {"type": "meta", ...}; every other line summarizes one output/rollouts/<run>/<timestamp>/<kind>/step_NNNN.jsonl.
Unchanged step files are not re-read: their entries are reused when the path, size and mtime match.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

SCHEMA = "grpo-rollouts-index-v1"
ROOT = Path(__file__).parent
ROLLOUTS = ROOT / "output" / "rollouts"
INDEX = ROOT / "output" / "rollouts-index.jsonl"


def summarize(path: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = max(len(rows), 1)
    rewards = [r["reward"] for r in rows]
    entry = {
        "count": len(rows),
        "accuracy": round(100 * sum(abs(x - 1.0) < 1e-8 for x in rewards) / n, 2),
        "partial": sum(abs(x - 0.1) < 1e-8 for x in rewards),
        "failed": sum(abs(x) < 1e-8 for x in rewards),
        "mean_reward": round(sum(rewards) / n, 4),
        "truncated": sum(r.get("finish_reason") == "length" for r in rows),
        "avg_tokens": round(sum(r.get("output_tokens", 0) for r in rows) / n, 1),
    }
    if rows and "group" in rows[0]:
        groups: Dict[int, set] = {}
        for r in rows:
            groups.setdefault(r["group"], set()).add(r["reward"])
        entry["groups"] = len(groups)
        entry["flat_groups"] = sum(len(v) == 1 for v in groups.values())  # zero advantage: no gradient from the group
    return entry


def refresh() -> bool:
    previous: Dict[str, Dict[str, Any]] = {}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "entry":
                previous[row["path"]] = row
    entries: List[Dict[str, Any]] = []
    for path in sorted(ROLLOUTS.glob("*/*/*/step_*.jsonl")):
        run, timestamp, kind = path.parts[-4:-1]
        stat = path.stat()
        key = path.relative_to(ROOT).as_posix()
        old = previous.get(key)
        if old and old["size"] == stat.st_size and old["mtime"] == int(stat.st_mtime):
            entries.append(old)
            continue
        entries.append({"type": "entry", "path": key, "run": run, "timestamp": int(timestamp), "kind": kind,
                        "step": int(path.stem.split("_")[1]), "size": stat.st_size, "mtime": int(stat.st_mtime),
                        **summarize(path)})
    if [e["path"] for e in entries] == list(previous) and all(e is previous[e["path"]] for e in entries):
        return False
    meta = {"type": "meta", "schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": len(entries)}
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in [meta, *entries]), encoding="utf-8")
    os.replace(tmp, INDEX)
    return True


if __name__ == "__main__":
    print(f"{'wrote' if refresh() else 'unchanged'} {INDEX.relative_to(ROOT)}")
