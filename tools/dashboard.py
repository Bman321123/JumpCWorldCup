"""Barebones localhost dashboard for the optimal odds (ROADMAP 3.1).

  python tools/dashboard.py        # http://127.0.0.1:8770

Lists tonight's matches from the platform, and on demand prices one match
through the full pipeline (model + scraped sharp odds + crowd + policy),
showing model / market / crowd / SUBMIT / confidence side by side. Pure stdlib
HTTP server; results cached in-process for 5 minutes per match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auto_trader import load_criteria, plan_submissions       # noqa: E402
from src.crowd_capture import fuzzy_lookup, latest_crowd          # noqa: E402
from src.orchestrator import Orchestrator                         # noqa: E402
from src.platform_client import PlatformClient, to_platform_probability  # noqa: E402
from src.submission_policy import submission                      # noqa: E402

LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
DB = str(ROOT / "data" / "wc_forecasting.db")
CACHE_TTL = 300
_cache: dict = {}

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Probability Cup — Optimal Odds</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0b1220;color:#e2e8f0;margin:0;padding:24px}
 h1{font-size:18px;color:#4f8cff;margin:0 0 4px}
 .sub{color:#64748b;font-size:12px;margin-bottom:18px}
 .matches{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
 .m{background:#111c33;border:1px solid #1e3a5f;border-radius:8px;padding:8px 12px;
    cursor:pointer;font-size:13px}
 .m:hover{border-color:#4f8cff}
 .m .t{color:#94a3b8;font-size:11px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #1a2942}
 th{color:#64748b;font-weight:600;font-size:11px;text-transform:uppercase}
 td.n{text-align:right;font-variant-numeric:tabular-nums}
 .submit{font-weight:700;color:#7dd3fc}
 .edge-hi{color:#4ade80;font-weight:700}
 .auto{color:#4ade80} .hand{color:#fbbf24}
 .flag{color:#f87171;font-size:11px}
 .bar{display:inline-block;height:8px;border-radius:2px;background:#334155;vertical-align:middle}
 .barf{display:inline-block;height:8px;border-radius:2px;background:#4f8cff;vertical-align:middle}
 #status{color:#64748b;font-size:12px;margin:8px 0}
</style></head><body>
<h1>Probability Cup — Optimal Odds</h1>
<div class="sub">model vs sharp market vs crowd, with the policy-adjusted SUBMIT value.
 Click a match to price it. (Pricing scrapes live odds — ~20s.)</div>
<div class="matches" id="matches">loading matches…</div>
<div id="status"></div>
<div id="table"></div>
<script>
async function loadMatches(){
 const r = await fetch('/api/matches'); const ms = await r.json();
 document.getElementById('matches').innerHTML = ms.map(m =>
  `<div class="m" onclick="price('${m.name}','${m.home}','${m.away}','${m.date}')">
    ${m.name}<div class="t">closes ${m.closes||''}</div></div>`).join('');
}
async function price(name,home,away,date){
 document.getElementById('status').textContent = 'Pricing '+name+' … scraping sharp odds, ~20s';
 document.getElementById('table').innerHTML='';
 const r = await fetch(`/api/edge?home=${home}&away=${away}&date=${date}`);
 const d = await r.json();
 if(d.error){document.getElementById('status').textContent='Error: '+d.error;return;}
 document.getElementById('status').textContent =
   name+'  ·  lambdas '+d.lam_home+' / '+d.lam_away+'  ·  '+(d.market?'market live':'model only');
 const rows = d.rows.map(x=>{
   const crowd = x.crowd!=null?(x.crowd*100).toFixed(0)+'%':'—';
   const mkt = x.market!=null?(x.market*100).toFixed(0)+'%':'—';
   const edge = x.edge!=null?(x.edge*100).toFixed(0):'—';
   const eclass = (x.edge!=null&&x.edge>0.10)?'edge-hi':'';
   const w = Math.round((x.confidence||0)*60);
   return `<tr>
     <td>${x.question}${x.flag?' <span class="flag">'+x.flag+'</span>':''}</td>
     <td class="n">${(x.model*100).toFixed(0)}%</td>
     <td class="n">${mkt}</td>
     <td class="n">${crowd}</td>
     <td class="n submit">${x.submit}%</td>
     <td class="n ${eclass}">${edge}</td>
     <td><span class="barf" style="width:${w}px"></span><span class="bar" style="width:${60-w}px"></span>
         <span class="${x.auto?'auto':'hand'}"> ${x.auto?'AUTO':'hand'}</span></td>
   </tr>`;}).join('');
 document.getElementById('table').innerHTML =
   `<table><tr><th>question</th><th>model</th><th>market</th><th>crowd</th>
    <th>submit</th><th>edge</th><th>confidence</th></tr>${rows}</table>`;
}
loadMatches();
</script></body></html>"""


def _codes():
    with open(ROOT / "config" / "groups.json") as f:
        teams = json.load(f)["teams"]
    idx = {}
    for code, t in teams.items():
        idx[code.lower()] = code
        idx[t["name"].lower()] = code
        for a in t.get("aliases", []):
            idx[a.lower()] = code
    return idx


class Handler(BaseHTTPRequestHandler):
    orch = None
    idx = {}
    crit = {}

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):                            # noqa: N802
        path = urlparse(self.path)
        if path.path == "/":
            return self._send(200, PAGE, "text/html")
        if path.path == "/api/matches":
            return self._send(200, json.dumps(self._matches()))
        if path.path == "/api/edge":
            q = parse_qs(path.query)
            return self._send(200, json.dumps(self._edge(
                q.get("home", [""])[0], q.get("away", [""])[0],
                q.get("date", [""])[0])))
        return self._send(404, json.dumps({"error": "not found"}))

    def _matches(self):
        try:
            client = PlatformClient()
            out = []
            for m in client.list_matches(lobby_id=LOBBY_ID):
                parts = m["name"].replace(" vs ", "|").split("|")
                if len(parts) != 2:
                    continue
                out.append({
                    "name": m["name"],
                    "home": self.idx.get(parts[0].strip().lower(),
                                         parts[0].strip().upper()[:3]),
                    "away": self.idx.get(parts[1].strip().lower(),
                                         parts[1].strip().upper()[:3]),
                    "date": (m.get("opening_time") or "")[:10],
                    "closes": (m.get("closing_time") or "")[11:16]})
            return out
        except Exception as e:                   # noqa: BLE001
            return [{"name": f"(platform error: {e})", "home": "", "away": "",
                     "date": "", "closes": ""}]

    def _edge(self, home, away, date):
        key = f"{home}:{away}:{date}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        try:
            client = PlatformClient()
            match = next((m for m in client.list_matches(lobby_id=LOBBY_ID)
                          if home in m["name"].upper() or away in m["name"].upper()),
                         None)
            if not match:
                return {"error": "match not found on platform"}
            markets = client.list_markets(LOBBY_ID, match["id"])
            questions = [mk.get("question") or mk.get("title") for mk in markets]
            manifest = self.orch.predict_match(home, away, date, questions, "group")
            crowd = latest_crowd(DB) if Path(DB).exists() else {}
            submit_values = {}
            rows = []
            for mk, pred in zip(markets, manifest["predictions"]):
                hitc = fuzzy_lookup(pred["question_text"], crowd)
                crowd_p = hitc["crowd_pct"] / 100.0 if hitc else None
                sv = to_platform_probability(submission(
                    pred["final_probability"], crowd_p,
                    pred["question_family"], "neutral"))
                submit_values[pred["question_id"]] = sv
                rows.append({"question": pred["question_text"],
                             "model": pred["model_probability"],
                             "market": pred["market_probability"],
                             "crowd": crowd_p, "submit": sv,
                             "edge": (abs(pred["final_probability"] - crowd_p)
                                      if crowd_p is not None else None),
                             "flag": ("FALLBACK" if pred["source"] == "fallback"
                                      else "")})
            decisions = {d.question_id: d for d in
                         plan_submissions(manifest, submit_values, self.crit)}
            for row, pred in zip(rows, manifest["predictions"]):
                d = decisions[pred["question_id"]]
                row["confidence"] = d.confidence
                row["auto"] = d.auto_eligible
            result = {"lam_home": manifest["model_params"]["lambda_home"],
                      "lam_away": manifest["model_params"]["lambda_away"],
                      "market": manifest["market_available"], "rows": rows}
            _cache[key] = (time.time(), result)
            return result
        except Exception as e:                   # noqa: BLE001
            return {"error": str(e)}

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    Handler.idx = _codes()
    Handler.crit = load_criteria(str(ROOT / "config" / "auto_trade.json"))
    Handler.orch = Orchestrator(
        config_dir=str(ROOT / "config"),
        params_path=str(ROOT / "params" / "dixon_coles.json"),
        db_path=DB if Path(DB).exists() else None,
        player_shares_path=str(ROOT / "config" / "player_shares.json"),
        online=True)
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
