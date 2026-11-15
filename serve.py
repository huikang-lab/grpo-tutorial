"""
Serve the rollout viewer (stdlib only).

    python serve.py              # then open http://127.0.0.1:44834/rollouts.html
    python serve.py --port 8000

Files are served from this folder. A top-level name this folder does not have falls through to viewer/<name>.
Loading output/rollouts-index.jsonl first refreshes it, so new training steps show up on reload.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rollouts_index

ROOT = Path(__file__).parent
VIEWER = ROOT / "viewer"


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        local = Path(super().translate_path(path))
        name = local.relative_to(ROOT).as_posix() if local.is_relative_to(ROOT) else ""
        if name == "output/rollouts-index.jsonl":
            rollouts_index.refresh()
        if name and "/" not in name and not local.exists() and (VIEWER / name).exists():
            return str(VIEWER / name)
        return str(local)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=44834)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"http://127.0.0.1:{args.port}/rollouts.html")
    server.serve_forever()
