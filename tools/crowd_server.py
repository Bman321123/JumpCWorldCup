"""Local receiver for the crowd-capture browser extension.

Run it whenever you're on the platform:
    python tools/crowd_server.py            # listens on 127.0.0.1:8765

The extension POSTs captured question blocks here; parsed questions + crowd
percentages land in the crowd_capture table, raw payloads in data/crowd_raw/
(nothing is ever discarded — formats can be re-parsed later).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crowd_capture import parse_blocks, store_capture  # noqa: E402

DB_PATH = str(ROOT / "data" / "wc_forecasting.db")
RAW_DIR = ROOT / "data" / "crowd_raw"


class Handler(BaseHTTPRequestHandler):
    db_path = DB_PATH

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Chrome Private Network Access preflight (https page -> localhost)
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):                        # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):                            # noqa: N802
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                           # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            blocks = payload.get("blocks", [])
            url = payload.get("url", "")

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            (RAW_DIR / f"{stamp}.json").write_text(json.dumps(payload, indent=1))

            rows = parse_blocks(blocks)
            n = store_capture(self.db_path, rows, url)
            for r in rows:
                flag = " (AMBIGUOUS)" if r["ambiguous"] else ""
                print(f"  [{r['crowd_pct']:5.1f}%]{flag} {r['question_text']}")
            print(f"Captured {n} question(s) from {len(blocks)} block(s) <- {url}")

            body = json.dumps({"stored": n, "blocks": len(blocks)}).encode()
            self.send_response(200)
        except Exception as e:                   # noqa: BLE001
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):                # quiet default access log
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    Handler.db_path = args.db
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    print(f"Crowd capture server on http://127.0.0.1:{args.port}  (db: {args.db})")
    print("Open the platform page with the extension installed; captures appear here.")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
