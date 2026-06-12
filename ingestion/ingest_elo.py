"""Fetch World Football Elo ratings (eloratings.net) -> params/elo.json.

Joins World.tsv (col 2 = eloratings 2-letter code, col 3 = rating) with
en.teams.tsv (code -> name + alias columns). Output is keyed by every name
variant so the Dixon-Coles prior lookup matches martj42 spellings directly.
Used as shrinkage priors in the fit (PRD v2.2 §4.2); on any failure the fit
simply runs without priors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RATINGS_URL = "https://www.eloratings.net/World.tsv"
TEAMS_URL = "https://www.eloratings.net/en.teams.tsv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; polite)"}


def _get(url: str) -> str:
    import requests
    r = requests.get(url, timeout=10, headers=HEADERS)
    r.raise_for_status()
    return r.text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "params" / "elo.json"))
    args = ap.parse_args()
    try:
        ratings_tsv = _get(RATINGS_URL)
        teams_tsv = _get(TEAMS_URL)
    except Exception as e:                       # noqa: BLE001
        print(f"Elo fetch failed ({e}); skipping — fit will run without priors.")
        sys.exit(0)

    code_to_names: dict[str, list[str]] = {}
    for line in teams_tsv.splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 2:
            code_to_names[parts[0]] = parts[1:]

    elo: dict[str, float] = {}
    for line in ratings_tsv.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code = parts[2].strip()
        try:
            rating = float(parts[3])
        except ValueError:
            continue
        if not (800 <= rating <= 2400):
            continue
        for name in code_to_names.get(code, []):
            elo[name] = rating

    if len(elo) < 100:
        print(f"Parsed only {len(elo)} ratings; format may have changed. Skipping.")
        sys.exit(0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(elo, f, indent=1, sort_keys=True, ensure_ascii=False)
    print(f"Saved {len(elo)} Elo name->rating entries to {args.out}")
    for t in ("Spain", "Argentina", "France", "Mexico", "Curaçao", "Jordan", "Haiti"):
        print(f"  {t:10s} {elo.get(t, 'MISSING')}")


if __name__ == "__main__":
    main()
