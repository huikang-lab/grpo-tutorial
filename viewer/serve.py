"""
The repository server: pages from viewer/, everything else from the root (stdlib only).

    ./serve.sh                                  # starts this with the index watcher, then prints every page's URL
    python viewer/serve.py --port 44834 --root .

The viewers live in this directory but serve under their bare names (/training.html, /metrics.html). A top-level name
the root has itself (the indexes, index.html) stays the root's; only what the root does not have falls through to viewer/.
"""
import argparse
import posixpath
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ViewerHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        resolved = super().translate_path(path)
        bare = path.split("?", 1)[0].split("#", 1)[0]
        name = posixpath.basename(bare)
        if bare.strip("/") == name and name and not Path(resolved).exists():
            candidate = Path(self.directory) / "viewer" / name
            if candidate.is_file():
                return str(candidate)
        return resolved

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), partial(ViewerHandler, directory=args.root))
    server.serve_forever()


if __name__ == "__main__":
    main()
