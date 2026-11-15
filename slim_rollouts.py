"""Rewrite rollout step files to the slim schema: token ids and log-probs, with the text left to the tokenizer.

The viewer decodes `tokens` and `prompt_tokens` with the committed tokenizer/tokenizer.json, as arc3-coach does, so a
row no longer carries `text`, `pieces` or `prompt_pieces`. A training group's prompt is kept on its sample 0 only.
Rewriting a file that is already slim leaves it unchanged.

    python slim_rollouts.py output/rollouts
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

DERIVED = ("text", "pieces", "prompt_pieces")


def slim_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: v for k, v in row.items() if k not in DERIVED}
    if "prompt_tokens" in row:
        row.pop("prompt", None)  # the ids decode to the same text
    if row.get("sample", 0) != 0:
        row.pop("prompt", None)
        row.pop("prompt_tokens", None)
    return row


def slim_text(text: str) -> str:
    rows = [slim_row(json.loads(line)) for line in text.splitlines() if line.strip()]
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def main() -> None:
    before = after = 0
    for path in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "output/rollouts").rglob("step_*.jsonl")):
        text = path.read_text(encoding="utf-8")
        slim = slim_text(text)
        before, after = before + len(text.encode()), after + len(slim.encode())
        if slim != text:
            path.with_suffix(".jsonl.tmp").write_text(slim, encoding="utf-8")
            os.replace(path.with_suffix(".jsonl.tmp"), path)
    print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
