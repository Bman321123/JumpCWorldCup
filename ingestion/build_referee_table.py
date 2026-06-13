"""Build config/referee_table.json from ESPN match referee + card data (ROADMAP B).

Aggregates yellows/reds per match by referee across the corpus. The multiplier
the model applies is shrunk toward 1.0 below ~10 matches (handled in
context_resolver.RefereeTable), so even a referee with 3 logged matches is safe.

  python ingestion/build_referee_table.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "data" / "match_stats"


def norm(name: str) -> str:
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def main() -> None:
    refs: dict = defaultdict(lambda: {"name": "", "matches": 0,
                                      "yellows": 0, "reds": 0})
    for f in sorted(STATS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        ref = data.get("referee")
        teams = data.get("teams") or {}
        if not ref or len(teams) != 2:
            continue
        y = sum(t.get("yellows", 0) for t in teams.values())
        r = sum(t.get("reds", 0) for t in teams.values())
        acc = refs[norm(ref)]
        acc["name"] = ref
        acc["matches"] += 1
        acc["yellows"] += y
        acc["reds"] += r

    out = {}
    for key, acc in refs.items():
        m = acc["matches"]
        out[key] = {"name": acc["name"],
                    "yellow_per_match": round(acc["yellows"] / m, 3),
                    "red_per_match": round(acc["reds"] / m, 3),
                    "total_matches": m}

    path = ROOT / "config" / "referee_table.json"
    path.write_text(json.dumps({"_source": "ESPN match officials + card counts",
                                "referees": out}, indent=1))
    print(f"referee_table.json: {len(out)} referees")
    for key, v in sorted(out.items(), key=lambda kv: -kv[1]["total_matches"])[:12]:
        print(f"  {v['name']:24s}  {v['yellow_per_match']:.2f} Y/m  "
              f"{v['red_per_match']:.2f} R/m  n={v['total_matches']}")


if __name__ == "__main__":
    main()
