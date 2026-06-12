"""Pull settled results (with Brier scores) from the platform into
predictions_log — closes the calibration loop automatically (ROADMAP 4).

  python tools/sync_results.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowd_capture import normalize_question      # noqa: E402
from src.platform_client import PlatformClient        # noqa: E402

LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    args = ap.parse_args()

    client = PlatformClient()
    results = client.list_results(LOBBY_ID)
    raw_dir = ROOT / "data" / "platform_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (raw_dir / f"{stamp}.json").write_text(json.dumps(results, indent=1))

    if not isinstance(results, list):
        print("Unexpected results payload — saved raw, nothing synced.")
        return

    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT prediction_id, question_text FROM predictions_log "
                       "WHERE actual_outcome IS NULL").fetchall()
    by_norm = {normalize_question(text): pid for pid, text in rows if text}

    synced, brier_sum = 0, 0.0
    for r in results:
        if not isinstance(r, dict):
            continue
        question = (r.get("question") or r.get("market_question")
                    or r.get("title") or "")
        outcome = r.get("outcome")
        if outcome is None and r.get("result") is not None:
            outcome = r["result"]
        brier = r.get("brier_score")
        pid = by_norm.get(normalize_question(question)) if question else None
        if pid and (outcome is not None or brier is not None):
            con.execute(
                "UPDATE predictions_log SET actual_outcome = ?, "
                "brier_contribution = ? WHERE prediction_id = ?",
                (None if outcome is None else int(bool(outcome)),
                 brier, pid))
            synced += 1
        if brier is not None:
            brier_sum += float(brier)
    con.commit()
    con.close()
    print(f"{len(results)} settled prediction(s) on platform; "
          f"{synced} matched into predictions_log; raw saved to {raw_dir}/")
    if results:
        print(f"Mean platform Brier so far: {brier_sum / len(results):.4f}")


if __name__ == "__main__":
    main()
