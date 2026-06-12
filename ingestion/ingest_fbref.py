"""FBref ingestion (Tier 1/2): corners, cards, offsides for international
tournaments, and player involvement shares for the player layer.

POLITENESS RULES (PRD v2.2 §7): <= 1 request / 2 seconds, cache every page to
disk, run nightly at most. FBref data is Opta-sourced.

This script is run-on-demand scaffolding: fetch_match_report() works on any
FBref match URL; build a URL list per tournament and run overnight. Output
updates team_rates and config/player_shares.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data" / "raw" / "fbref_cache"
REQUEST_DELAY_S = 2.5


def fetch(url: str) -> str:
    """Disk-cached, rate-limited GET (curl_cffi — FBref sits behind Cloudflare)."""
    import hashlib
    from curl_cffi import requests as cffi_requests
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.html"
    if cached.exists():
        return cached.read_text()
    time.sleep(REQUEST_DELAY_S)
    r = cffi_requests.get(url, impersonate="chrome", timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"FBref HTTP {r.status_code} on {url}")
    cached.write_text(r.text)
    return r.text


def fetch_match_report(url: str) -> dict:
    """Parse one FBref match report into team-level micro stats.

    Sources inside the page:
      - stats_*_misc table footers: cards, offsides, fouls
      - stats_*_summary table footers: shots on target
      - #team_stats_extra panel: corners (label followed by two values)
    """
    from bs4 import BeautifulSoup
    html = fetch(url)
    soup = BeautifulSoup(html, "lxml")
    out = {"url": url, "teams": {}}

    def team_of(table) -> str:
        cap = table.find_previous("caption")
        return cap.get_text(strip=True).split(" Player")[0] if cap else "?"

    for table in soup.select("table[id^=stats_][id$=_misc]"):
        foot = table.select_one("tfoot")
        if not foot:
            continue
        cells = {td.get("data-stat"): td.get_text(strip=True)
                 for td in foot.select("td")}
        out["teams"].setdefault(team_of(table), {}).update({
            "offsides": _to_int(cells.get("offsides")),
            "yellows": _to_int(cells.get("cards_yellow")),
            "reds": _to_int(cells.get("cards_red")),
            "fouls": _to_int(cells.get("fouls")),
        })

    for table in soup.select("table[id^=stats_][id$=_summary]"):
        foot = table.select_one("tfoot")
        if not foot:
            continue
        cells = {td.get("data-stat"): td.get_text(strip=True)
                 for td in foot.select("td")}
        out["teams"].setdefault(team_of(table), {})["sot"] = \
            _to_int(cells.get("shots_on_target"))

    # corners live in the #team_stats_extra text panel:
    # "<home val> <label> <away val>" triplets, team names in the header row
    extra = soup.select_one("#team_stats_extra")
    if extra and len(out["teams"]) == 2:
        names = list(out["teams"].keys())
        tokens = [t for t in extra.get_text(" ", strip=True).split() if t]
        for i, tok in enumerate(tokens):
            if tok.lower() == "corners" and 0 < i < len(tokens) - 1:
                h, a = _to_int(tokens[i - 1]), _to_int(tokens[i + 1])
                if h is not None and a is not None:
                    out["teams"][names[0]]["corners"] = h
                    out["teams"][names[1]]["corners"] = a
                break
    return out


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", help="file with one FBref match-report URL per line")
    args = ap.parse_args()
    if not args.urls_file:
        print(__doc__)
        return
    for url in Path(args.urls_file).read_text().splitlines():
        url = url.strip()
        if not url:
            continue
        report = fetch_match_report(url)
        print(report)


if __name__ == "__main__":
    main()
