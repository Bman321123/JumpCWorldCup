"""Scrape FBref tournament match reports -> per-team micro rates (ROADMAP 1.1).

Collects match-report URLs from competition schedule pages, fetches each
report politely (2.5s delay, disk cache), saves parsed team stats to
data/fbref_parsed/, then aggregates into per-team rates and writes them into
params/dixon_coles.json (keyed by team name AND FIFA code).

  python ingestion/run_fbref_scrape.py                  # default recent majors
  python ingestion/run_fbref_scrape.py --comps WC2022   # single tournament
  python ingestion/run_fbref_scrape.py --aggregate-only # recompute rates only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingestion.ingest_fbref import fetch, fetch_match_report  # noqa: E402

PARSED_DIR = ROOT / "data" / "match_stats"   # shared with ingest_espn.py

COMPS = {
    "WC2022": "https://fbref.com/en/comps/1/2022/schedule/2022-FIFA-World-Cup-Scores-and-Fixtures",
    "WC2018": "https://fbref.com/en/comps/1/2018/schedule/2018-FIFA-World-Cup-Scores-and-Fixtures",
    "EURO2024": "https://fbref.com/en/comps/676/2024/schedule/2024-European-Championship-Scores-and-Fixtures",
    "COPA2024": "https://fbref.com/en/comps/685/2024/schedule/2024-Copa-America-Scores-and-Fixtures",
    "WC2026": "https://fbref.com/en/comps/1/schedule/FIFA-World-Cup-Scores-and-Fixtures",
}
MATCH_HREF = re.compile(r"/en/matches/[0-9a-f]{8}/[A-Za-z0-9\-]+")
SHRINK_N = 8.0      # pseudo-matches toward the global mean
GLOBAL = {"corners": 4.9, "yellows": 1.7, "reds": 0.09, "offsides": 2.0, "sot": 4.3}


def collect_urls(schedule_url: str) -> list:
    html = fetch(schedule_url)
    urls = sorted({f"https://fbref.com{m}" for m in MATCH_HREF.findall(html)})
    return urls


def scrape(comps: list) -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    for comp in comps:
        url = COMPS.get(comp)
        if not url:
            print(f"Unknown comp {comp}; options: {list(COMPS)}")
            continue
        try:
            urls = collect_urls(url)
        except Exception as e:                   # noqa: BLE001
            print(f"{comp}: schedule fetch failed: {e}")
            continue
        print(f"{comp}: {len(urls)} match reports")
        for i, u in enumerate(urls):
            slug = u.rsplit("/", 1)[-1]
            out_path = PARSED_DIR / f"{comp}_{slug}.json"
            if out_path.exists():
                continue
            try:
                report = fetch_match_report(u)
                if report["teams"]:
                    out_path.write_text(json.dumps(report, indent=1))
            except Exception as e:               # noqa: BLE001
                print(f"  {slug}: {e}")
            if (i + 1) % 20 == 0:
                print(f"  {comp}: {i + 1}/{len(urls)}")


def aggregate() -> None:
    """Parsed reports -> shrunk per-team rates -> params/dixon_coles.json."""
    from src.stats_engine import ModelParameters
    sums: dict = {}
    counts: dict = {}
    against: dict = {}
    for f in PARSED_DIR.glob("*.json"):
        report = json.loads(f.read_text())
        teams = list(report.get("teams", {}).items())
        if len(teams) != 2:
            continue
        for (name, stats), (_, opp_stats) in (
                (teams[0], teams[1]), (teams[1], teams[0])):
            t = sums.setdefault(name, {k: 0.0 for k in GLOBAL})
            counts[name] = counts.get(name, 0) + 1
            for k in GLOBAL:
                if stats.get(k) is not None:
                    t[k] += stats[k]
            if opp_stats.get("corners") is not None:
                against[name] = against.get(name, 0.0) + opp_stats["corners"]

    params = ModelParameters.load(str(ROOT / "params" / "dixon_coles.json"))
    with open(ROOT / "config" / "groups.json") as f:
        teams_cfg = json.load(f)["teams"]
    alias_to_code = {}
    for code, t in teams_cfg.items():
        for cand in [t["name"]] + t.get("aliases", []):
            alias_to_code[cand.lower()] = code

    updated = 0
    for name, n in counts.items():
        if n < 2:
            continue
        keys = [name]
        code = alias_to_code.get(name.lower())
        if code:
            keys.append(code)
        for k, table_name in (("corners", "corner_for"), ("yellows", "yellow_rates"),
                              ("reds", "red_rates"), ("offsides", "offside_rates"),
                              ("sot", "sot_rates")):
            rate = (sums[name][k] + GLOBAL[k] * SHRINK_N) / (n + SHRINK_N)
            for key in keys:
                getattr(params, table_name)[key] = round(rate, 3)
        ca = (against.get(name, 0.0) + GLOBAL["corners"] * SHRINK_N) / (n + SHRINK_N)
        for key in keys:
            params.corner_against[key] = round(ca, 3)
        updated += 1
    params.save(str(ROOT / "params" / "dixon_coles.json"))
    wc_covered = sum(1 for c in teams_cfg if c in params.corner_for)
    print(f"Micro rates updated for {updated} teams "
          f"({wc_covered}/48 WC teams covered) -> params/dixon_coles.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="WC2022,EURO2024,COPA2024,WC2026")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        scrape([c.strip() for c in args.comps.split(",")])
    aggregate()
