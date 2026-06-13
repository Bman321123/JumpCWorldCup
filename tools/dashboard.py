"""Localhost control panel for the Probability Cup (ROADMAP 3.1 + submission).

  python tools/dashboard.py          # http://127.0.0.1:8770

Shows optimal odds per match (model / market / crowd / SUBMIT / edge /
confidence), and lets you act:
  - "Submit all" / "Submit auto-eligible": YOU click, it posts to the platform
    via the bot API (manual submission you drive).
  - Autopilot arm toggle: flips config/auto_trade.json. When armed, the
    autopilot button submits auto-eligible questions across matches.

Safety: every submission is an explicit button click here. Claude does not
arm autopilot or click submit — those are your actions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
AUTO_CFG = ROOT / "config" / "auto_trade.json"
_cache: dict = {}

PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Probability Cup — Control Panel</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0b1220;color:#e2e8f0;margin:0;padding:24px}
 h1{font-size:18px;color:#4f8cff;margin:0 0 4px}
 .sub{color:#64748b;font-size:12px;margin-bottom:14px}
 .bar{display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap}
 .pill{background:#111c33;border:1px solid #1e3a5f;border-radius:20px;padding:5px 12px;font-size:12px}
 .armed{border-color:#7f1d1d;color:#fca5a5} .disarmed{border-color:#14532d;color:#86efac}
 .matches{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
 .m{background:#111c33;border:1px solid #1e3a5f;border-radius:8px;padding:8px 12px;cursor:pointer;font-size:13px}
 .m:hover{border-color:#4f8cff}.m .t{color:#94a3b8;font-size:11px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #1a2942}
 th{color:#64748b;font-weight:600;font-size:11px;text-transform:uppercase}
 td.n{text-align:right;font-variant-numeric:tabular-nums}
 .submit{font-weight:700;color:#7dd3fc}.edge-hi{color:#4ade80;font-weight:700}
 .auto{color:#4ade80}.hand{color:#fbbf24}.flag{color:#f87171;font-size:11px}
 button{background:#4f8cff;color:#fff;border:none;border-radius:7px;padding:7px 12px;font-weight:600;cursor:pointer;font-size:13px}
 button:hover{background:#3b7af0}button.warn{background:#b91c1c}button.warn:hover{background:#991b1b}
 button:disabled{background:#334155;cursor:not-allowed}
 #status{color:#94a3b8;font-size:12px;margin:10px 0}
 .barf{display:inline-block;height:8px;border-radius:2px;background:#4f8cff;vertical-align:middle}
 .barr{display:inline-block;height:8px;border-radius:2px;background:#334155;vertical-align:middle}
</style></head><body>
<h1>Probability Cup — Control Panel</h1>
<div class="sub">model vs sharp market vs crowd, with the policy SUBMIT value. You submit; nothing is sent without your click.</div>
<div class="bar">
 <span class="pill" id="armpill">autopilot: …</span>
 <button id="armbtn" onclick="toggleArm()">…</button>
 <button onclick="runAuto()" id="autobtn">Submit auto-eligible (all matches)</button>
</div>
<div class="card" style="background:#111c33;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:18px">
 <div style="font-weight:600;color:#7dd3fc;margin-bottom:8px">Custom questions (price any questions for any fixture)</div>
 <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">
  <input id="ch" value="GER" style="width:60px;background:#0b1220;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:6px">
  <input id="ca" value="CUW" style="width:60px;background:#0b1220;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:6px">
  <input id="cd" value="2026-06-13" style="width:120px;background:#0b1220;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:6px">
  <button onclick="priceCustom()">Price these</button>
  <button onclick="checkEvents()" style="background:#475569">Current events</button>
 </div>
 <div id="eventsresult" style="margin-bottom:8px"></div>
 <textarea id="cq" rows="3" style="width:100%;background:#0b1220;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:8px;font-size:13px">Will Curaçao commit more fouls than Germany?
Will Curaçao be caught offside 2 or more times?
In the second half, will Germany have more shots on target than Curaçao?</textarea>
 <div id="customresult" style="margin-top:10px"></div>
</div>
<div class="matches" id="matches">loading matches…</div>
<div id="status"></div>
<div id="actions"></div>
<div id="table"></div>
<script>
let CUR=null;
async function refreshStatus(){
 const s=await (await fetch('/api/status')).json();
 const armed=s.armed;
 document.getElementById('armpill').className='pill '+(armed?'armed':'disarmed');
 document.getElementById('armpill').textContent='autopilot: '+(armed?'ARMED':'disarmed');
 document.getElementById('armbtn').textContent=armed?'Disarm':'Arm autopilot';
 document.getElementById('armbtn').className=armed?'warn':'';
 document.getElementById('autobtn').disabled=!armed;
}
async function toggleArm(){
 const s=await (await fetch('/api/status')).json();
 const next=!s.armed;
 if(next && !confirm('ARM autopilot? When armed, the "Submit auto-eligible" button will post live bets to the platform without per-question review.')) return;
 await fetch('/api/arm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:next})});
 refreshStatus();
}
async function loadMatches(){
 const ms=await (await fetch('/api/matches')).json();
 document.getElementById('matches').innerHTML=ms.map(m=>
  `<div class="m" onclick="price('${m.name}','${m.home}','${m.away}','${m.date}')">${m.name}<div class="t">closes ${m.closes||''}</div></div>`).join('');
}
async function price(name,home,away,date){
 CUR={name,home,away,date};
 document.getElementById('status').textContent='Pricing '+name+' … scraping sharp odds (~20s)';
 document.getElementById('table').innerHTML='';document.getElementById('actions').innerHTML='';
 const d=await (await fetch(`/api/edge?home=${home}&away=${away}&date=${date}`)).json();
 if(d.error){document.getElementById('status').textContent='Error: '+d.error;return;}
 document.getElementById('status').textContent=name+'  ·  lambdas '+d.lam_home+' / '+d.lam_away+'  ·  '+(d.market?'market live':'model only');
 const autoN=d.rows.filter(r=>r.auto).length;
 document.getElementById('actions').innerHTML=
   `<button onclick="doSubmit('all')">Submit all ${d.rows.length} to platform</button>
    <button onclick="doSubmit('auto')">Submit ${autoN} auto-eligible</button>`;
 document.getElementById('table').innerHTML=
   `<table><tr><th>question</th><th>model</th><th>market</th><th>crowd</th><th>submit</th><th>edge</th><th>conf</th></tr>`+
   d.rows.map(x=>{
    const crowd=x.crowd!=null?(x.crowd*100).toFixed(0)+'%':'—';
    const mkt=x.market!=null?(x.market*100).toFixed(0)+'%':'—';
    const edge=x.edge!=null?(x.edge*100).toFixed(0):'—';
    const w=Math.round((x.confidence||0)*60);
    return `<tr><td>${x.question}${x.flag?' <span class="flag">'+x.flag+'</span>':''}</td>
     <td class="n">${(x.model*100).toFixed(0)}%</td><td class="n">${mkt}</td><td class="n">${crowd}</td>
     <td class="n submit">${x.submit}%</td><td class="n ${(x.edge>0.1?'edge-hi':'')}">${edge}</td>
     <td><span class="barf" style="width:${w}px"></span><span class="barr" style="width:${60-w}px"></span> <span class="${x.auto?'auto':'hand'}">${x.auto?'AUTO':'hand'}</span></td></tr>`;}).join('')+`</table>`;
}
async function doSubmit(which){
 if(!CUR)return;
 if(!confirm(`Submit ${which==='auto'?'auto-eligible':'ALL'} predictions for ${CUR.name} to the platform?`))return;
 document.getElementById('status').textContent='Submitting…';
 const r=await (await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...CUR,which})})).json();
 document.getElementById('status').textContent=r.error?('Error: '+r.error):(`Submitted ${r.submitted} prediction(s) for ${CUR.name}.`);
}
async function runAuto(){
 if(!confirm('Run autopilot across ALL open matches and submit every auto-eligible question now?'))return;
 document.getElementById('status').textContent='Autopilot running…';
 const r=await (await fetch('/api/autopilot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})).json();
 document.getElementById('status').textContent=r.error?('Error: '+r.error):(`Autopilot submitted ${r.submitted} prediction(s) across ${r.matches} match(es).`);
}
async function checkEvents(){
 const home=document.getElementById('ch').value.trim();
 const away=document.getElementById('ca').value.trim();
 const box=document.getElementById('eventsresult');
 box.innerHTML='<span style="color:#94a3b8">Checking status, lineup, news…</span>';
 const d=await (await fetch(`/api/events?home=${home}&away=${away}`)).json();
 if(d.error){box.innerHTML='<span style="color:#f87171">'+d.error+'</span>';return;}
 let h='<div style="font-size:12px;border-left:2px solid #475569;padding-left:8px">';
 if(d.status) h+=`<div>status: <b>${d.status.postponed?'<span style="color:#f87171">POSTPONED/DELAYED</span>':d.status.state}</b> · ${d.status.detail||''} · ${d.status.venue||''}</div>`;
 h+=`<div>lineup: ${d.lineup_published?('<b>published</b> — auto-applied'):'<span style="color:#94a3b8">not yet published</span>'}</div>`;
 if(d.absences && d.absences.length) h+=`<div style="color:#fbbf24">key absences: ${d.absences.join(', ')}</div>`;
 if(d.news && d.news.length){h+='<div style="margin-top:4px;color:#64748b">news (your judgment — not auto-applied):</div>';
   d.news.forEach(n=>h+=`<div style="color:#94a3b8">· ${n}</div>`);}
 box.innerHTML=h+'</div>';
}
async function priceCustom(){
 const home=document.getElementById('ch').value.trim();
 const away=document.getElementById('ca').value.trim();
 const date=document.getElementById('cd').value.trim();
 const questions=document.getElementById('cq').value.split('\n').map(s=>s.trim()).filter(s=>s);
 const box=document.getElementById('customresult');
 box.innerHTML='<span style="color:#94a3b8">Pricing through the full pipeline (~20s)…</span>';
 const d=await (await fetch('/api/custom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({home,away,date,questions})})).json();
 if(d.error){box.innerHTML='<span style="color:#f87171">Error: '+d.error+'</span>';return;}
 box.innerHTML=`<div style="color:#64748b;font-size:11px;margin-bottom:6px">${home} vs ${away} · lambdas ${d.lam_home}/${d.lam_away} · ${d.market?'market live':'model only'}</div>`+
  `<table><tr><th>question</th><th>probability</th><th>model</th><th>market</th><th>source</th></tr>`+
  d.rows.map(x=>{const mkt=x.market!=null?(x.market*100).toFixed(1)+'%':'—';
   return `<tr><td>${x.question}${x.flag?' <span class="flag">'+x.flag+'</span>':''}</td>
    <td class="n submit" style="font-size:15px">${(x.final*100).toFixed(1)}%</td>
    <td class="n">${(x.model*100).toFixed(1)}%</td><td class="n">${mkt}</td>
    <td>${x.source}</td></tr>`;}).join('')+`</table>`;
}
refreshStatus();loadMatches();
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

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):                            # noqa: N802
        p = urlparse(self.path)
        if p.path == "/":
            return self._send(200, PAGE, "text/html")
        if p.path == "/api/matches":
            return self._send(200, json.dumps(self._matches()))
        if p.path == "/api/status":
            return self._send(200, json.dumps({"armed": load_criteria(str(AUTO_CFG))["armed"]}))
        if p.path == "/api/edge":
            q = parse_qs(p.query)
            return self._send(200, json.dumps(self._edge(
                q.get("home", [""])[0], q.get("away", [""])[0], q.get("date", [""])[0])))
        if p.path == "/api/events":
            q = parse_qs(p.query)
            return self._send(200, json.dumps(self._events(
                q.get("home", [""])[0], q.get("away", [""])[0])))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):                           # noqa: N802
        p = urlparse(self.path)
        try:
            body = self._body()
            if p.path == "/api/arm":
                cfg = json.loads(AUTO_CFG.read_text())
                cfg["armed"] = bool(body.get("on"))
                AUTO_CFG.write_text(json.dumps(cfg, indent=1))
                return self._send(200, json.dumps({"armed": cfg["armed"]}))
            if p.path == "/api/submit":
                return self._send(200, json.dumps(self._submit(
                    body["home"], body["away"], body["date"],
                    body.get("which", "all"))))
            if p.path == "/api/autopilot":
                return self._send(200, json.dumps(self._autopilot()))
            if p.path == "/api/custom":
                return self._send(200, json.dumps(self._custom(
                    body["home"], body["away"], body["date"],
                    body.get("questions", []))))
        except Exception as e:                   # noqa: BLE001
            return self._send(200, json.dumps({"error": str(e)}))
        return self._send(404, json.dumps({"error": "not found"}))

    # ----- data ops -----

    def _matches(self):
        try:
            client = PlatformClient()
            out = []
            for m in client.list_matches(lobby_id=LOBBY_ID):
                parts = m["name"].replace(" vs ", "|").split("|")
                if len(parts) != 2:
                    continue
                out.append({"name": m["name"],
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

    def _price(self, home, away, date):
        """Returns (rows, manifest) with market_id + submit value + auto flag."""
        key = f"{home}:{away}:{date}"
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 300:
            return hit[1]
        client = PlatformClient()
        match = next((m for m in client.list_matches(lobby_id=LOBBY_ID)
                      if home in m["name"].upper() or away in m["name"].upper()), None)
        if not match:
            raise RuntimeError("match not found on platform")
        markets = client.list_markets(LOBBY_ID, match["id"])
        questions = [mk.get("question") or mk.get("title") for mk in markets]
        manifest = self.orch.predict_match(home, away, date, questions, "group")
        crowd = latest_crowd(DB) if Path(DB).exists() else {}
        submit_values, rows = {}, []
        for mk, pred in zip(markets, manifest["predictions"]):
            hitc = fuzzy_lookup(pred["question_text"], crowd)
            crowd_p = hitc["crowd_pct"] / 100.0 if hitc else None
            sv = to_platform_probability(submission(
                pred["final_probability"], crowd_p, pred["question_family"], "neutral"))
            submit_values[pred["question_id"]] = sv
            rows.append({"market_id": mk["id"], "question": pred["question_text"],
                         "model": pred["model_probability"],
                         "market": pred["market_probability"], "crowd": crowd_p,
                         "submit": sv,
                         "edge": (abs(pred["final_probability"] - crowd_p)
                                  if crowd_p is not None else None),
                         "flag": "FALLBACK" if pred["source"] == "fallback" else ""})
        decisions = {d.question_id: d for d in plan_submissions(
            manifest, submit_values, load_criteria(str(AUTO_CFG)))}
        for row, pred in zip(rows, manifest["predictions"]):
            d = decisions[pred["question_id"]]
            row["confidence"] = d.confidence
            row["auto"] = d.auto_eligible
        result = {"lam_home": manifest["model_params"]["lambda_home"],
                  "lam_away": manifest["model_params"]["lambda_away"],
                  "market": manifest["market_available"], "rows": rows}
        _cache[key] = (time.time(), result)
        return result

    def _edge(self, home, away, date):
        try:
            return self._price(home, away, date)
        except Exception as e:                   # noqa: BLE001
            return {"error": str(e)}

    def _submit(self, home, away, date, which):
        data = self._price(home, away, date)
        rows = [r for r in data["rows"] if which != "auto" or r["auto"]]
        payload = [{"market_id": r["market_id"], "lobby_id": LOBBY_ID,
                    "probability": r["submit"]} for r in rows]
        if not payload:
            return {"submitted": 0, "note": "nothing eligible"}
        result = PlatformClient().submit_batch(payload)
        return {"submitted": len(payload), "result": result}

    def _events(self, home, away):
        from src.live_context import (confirmed_xi, derive_absences,
                                      match_status, news_headlines)
        hn = self.orch.team_names.get(home, home)
        an = self.orch.team_names.get(away, away)
        xi = confirmed_xi(hn, an)
        absences = []
        if xi:
            absences = (derive_absences(home, xi.get("home", []),
                                        self.orch.players.players)
                        + derive_absences(away, xi.get("away", []),
                                          self.orch.players.players))
        return {"status": match_status(hn, an),
                "lineup_published": xi is not None,
                "absences": absences, "news": news_headlines(hn, an, 5)}

    def _custom(self, home, away, date, questions):
        """Price arbitrary questions for any fixture through the full pipeline."""
        if not questions:
            return {"error": "no questions provided"}
        manifest = self.orch.predict_match(home, away, date, questions, "group")
        crowd = latest_crowd(DB) if Path(DB).exists() else {}
        rows = []
        for pred in manifest["predictions"]:
            hitc = fuzzy_lookup(pred["question_text"], crowd)
            crowd_p = hitc["crowd_pct"] / 100.0 if hitc else None
            final = submission(pred["final_probability"], crowd_p,
                               pred["question_family"], "neutral")
            rows.append({"question": pred["question_text"],
                         "model": pred["model_probability"],
                         "market": pred["market_probability"],
                         "final": final,
                         "source": pred["source"],
                         "flag": "FALLBACK" if pred["source"] == "fallback" else ""})
        return {"lam_home": manifest["model_params"]["lambda_home"],
                "lam_away": manifest["model_params"]["lambda_away"],
                "market": manifest["market_available"], "rows": rows}

    def _autopilot(self):
        crit = load_criteria(str(AUTO_CFG))
        if not crit["armed"]:
            return {"error": "autopilot disarmed — arm it first"}
        client = PlatformClient()
        total, nmatch = 0, 0
        for m in client.list_matches(lobby_id=LOBBY_ID):
            parts = m["name"].replace(" vs ", "|").split("|")
            if len(parts) != 2:
                continue
            home = self.idx.get(parts[0].strip().lower(), parts[0].strip().upper()[:3])
            away = self.idx.get(parts[1].strip().lower(), parts[1].strip().upper()[:3])
            try:
                res = self._submit(home, away, (m.get("opening_time") or "")[:10], "auto")
                total += res.get("submitted", 0)
                nmatch += 1
            except Exception:                    # noqa: BLE001
                continue
        return {"submitted": total, "matches": nmatch}

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    Handler.idx = _codes()
    Handler.orch = Orchestrator(
        config_dir=str(ROOT / "config"),
        params_path=str(ROOT / "params" / "dixon_coles.json"),
        db_path=DB if Path(DB).exists() else None,
        player_shares_path=str(ROOT / "config" / "player_shares.json"),
        online=True)
    print(f"Control panel: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
