"""Live submission daemon — ONE sweep per invocation (run on a schedule).

Ensures every open market gets a prediction before kickoff and refreshes our own
predictions in the closing window with fresh lines + lineups. Submits EVERY open
market (the user's directive: answer all available markets), at our calibrated
number — never 50, clamped 1-99.

SAFETY (hardened after an adversarial review):
  - Submits ONLY when config/auto_trade.json "armed": true. Set false = instant
    kill switch (sweeps still run + log, but send nothing).
  - Idempotent over ALL prediction statuses (never double-submits).
  - Ownership journal (data/bot_submitted.json): the daemon only UPDATES
    predictions it created — it never overwrites a human's manual pick.
  - Fail-closed: if list_predictions/list_matches don't return a list, the sweep
    ABORTS rather than risk mass duplicate submission.
  - No fabricated team codes: an unresolvable team name SKIPS the match (logged).
  - Per-match isolation AND per-batch isolation: one failure never aborts the rest.
  - Lockfile prevents overlapping sweeps.
  - Round is inferred; unknown round (post-group, unmapped) SKIPS rather than
    misprice.

  python tools/live_submit.py          # run a sweep (submits iff armed)
  python tools/live_submit.py --dry    # never submits; logs what it would do
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auto_trader import load_criteria                        # noqa: E402
from src.crowd_capture import fuzzy_lookup, latest_crowd         # noqa: E402
from src.orchestrator import Orchestrator                        # noqa: E402
from src.platform_client import (PlatformClient,                 # noqa: E402
                                 to_platform_probability)
from src.schedule import infer_round                             # noqa: E402
from src.submission_policy import submission                     # noqa: E402

LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
DB = str(ROOT / "data" / "wc_forecasting.db")
AUTO_CFG = ROOT / "config" / "auto_trade.json"
LOG = ROOT / "data" / "live_submit.log"
JOURNAL = ROOT / "data" / "bot_submitted.json"
LOCK = ROOT / "data" / "live_submit.lock"

SUBMIT_WINDOW_H = 18.0     # start submitting once a match is within this window
CLOSING_WINDOW_H = 1.5     # always re-price + update OUR predictions inside this window
UPDATE_DELTA = 2           # outside the closing window, update only on a line move >= this many points
LOCK_STALE_S = 1800        # a lock older than this is considered abandoned
POSITION = "neutral"       # submit our honest number (crowd is opportunity, not anchor)

logger = logging.getLogger("live_submit")


def _log_setup():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(LOG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.handlers = [fh, sh]


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


def _resolve(name: str, idx: dict):
    """Return the FIFA code, or None on miss (NEVER fabricate a code)."""
    return idx.get(name.strip().lower())


NO_EDGE_SHRINK = 0.25      # how much of the model's deviation from 50 to KEEP


def _shrink_no_edge(p: float, question_text: str) -> float:
    """Neutralize corner/foul/card 'more than the opponent' comparatives toward 50.
    Measured on live results (n=25): these markets have NO edge — leave-one-out
    picks hug-50 over the model on every fold (Brier 0.329 -> 0.250). They are
    high-variance count differentials with little team-strength signal, so the
    model's confidence is pure noise that the quadratic punishes. SOT comparisons
    are deliberately NOT shrunk (they carry a real, measured edge)."""
    ql = question_text.lower()
    if "than" in ql and "shots on target" not in ql and (
            "more corner" in ql or "more foul" in ql or "more card" in ql
            or "receive more" in ql or "commit more" in ql):
        return 0.5 + (p - 0.5) * NO_EDGE_SHRINK
    return p


def _hours_to(iso) -> float:
    if not isinstance(iso, str):
        return 999.0
    try:
        ko = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return (ko - datetime.now(timezone.utc)).total_seconds() / 3600.0
    except Exception:                            # noqa: BLE001
        return 999.0


def _load_journal() -> dict:
    try:
        return json.loads(JOURNAL.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save_journal(j: dict):
    JOURNAL.write_text(json.dumps(j, indent=0))


def _acquire_lock() -> bool:
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_S:
            logger.info("another sweep holds the lock (%.0fs old); skipping.", age)
            return False
    LOCK.write_text(str(os.getpid()))
    return True


def _release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def sweep(dry: bool) -> dict:
    armed = bool(load_criteria(str(AUTO_CFG)).get("armed"))
    live = armed and not dry
    mode = "LIVE" if live else ("DRY (disarmed)" if not armed else "DRY (--dry)")
    idx = _codes()
    client = PlatformClient()

    preds = client.list_predictions(LOBBY_ID)
    if not isinstance(preds, list):
        logger.error("list_predictions did not return a list (%r) — ABORT sweep "
                     "(fail-closed to avoid double-submission).", str(preds)[:100])
        return {"aborted": True}
    # dedupe over ALL statuses, not just open
    existing = {p["market_id"]: p for p in preds
                if isinstance(p, dict) and p.get("market_id")}
    journal = _load_journal()           # market_ids the bot itself submitted

    matches = client.list_matches(lobby_id=LOBBY_ID)
    if not isinstance(matches, list):
        logger.error("list_matches did not return a list — ABORT sweep.")
        return {"aborted": True}

    logger.info("=== sweep %s | %d matches | %d existing predictions | %d bot-owned ===",
                mode, len(matches), len(existing), len(journal))

    orch = Orchestrator(config_dir=str(ROOT / "config"),
                        params_path=str(ROOT / "params" / "dixon_coles.json"),
                        db_path=DB, online=True,
                        player_shares_path=str(ROOT / "config" / "player_shares.json"))
    crowd = latest_crowd(DB) if Path(DB).exists() else {}

    new_payload, updates = [], []        # updates: (prediction_id, market_id, sv)
    s = {"priced": 0, "submitted": 0, "updated": 0, "errors": 0, "skipped": 0}

    for m in matches:
        try:
            h = _hours_to(m.get("opening_time"))
            if h > SUBMIT_WINDOW_H or h < -0.15:
                continue
            date = (m.get("opening_time") or "")[:10]
            rnd = infer_round(m.get("opening_time"))
            if rnd is None:
                logger.warning("  %s: kickoff %r not placeable in the tournament — "
                               "SKIP.", m.get("name"), m.get("opening_time"))
                s["skipped"] += 1
                continue
            parts = m.get("name", "").replace(" vs ", "|").split("|")
            if len(parts) != 2:
                continue
            home, away = _resolve(parts[0], idx), _resolve(parts[1], idx)
            if not home or not away:
                logger.error("  %s: unresolved team code (%s/%s) — SKIP.",
                             m.get("name"), parts[0], parts[1])
                s["skipped"] += 1
                continue
            markets = [mk for mk in client.list_markets(LOBBY_ID, m["id"])
                       if mk.get("status", "open") == "open" and mk.get("id")]
            # answered = in list_predictions OR in our journal (covers read-after-
            # write lag where a just-submitted market hasn't appeared yet)
            unsub = [mk for mk in markets
                     if mk["id"] not in existing and mk["id"] not in journal]
            near = h <= CLOSING_WINDOW_H
            # markets we own and can re-price as the line moves (full pre-kickoff
            # window, not just the closing window — this is the live market tracking)
            owned_open = [mk for mk in markets
                          if mk["id"] in existing and mk["id"] in journal]
            if not unsub and not owned_open:
                continue

            questions = [mk.get("question") or mk.get("title") or "" for mk in markets]
            manifest = orch.predict_match(home, away, date, questions, rnd)
            if len(manifest["predictions"]) != len(markets):
                logger.error("  %s: pred/market length mismatch — SKIP.", m.get("name"))
                s["errors"] += 1
                continue
            s["priced"] += 1

            n_new = n_upd = 0
            for mk, pred in zip(markets, manifest["predictions"]):
                fp = pred.get("final_probability")
                if fp is None:
                    continue
                hitc = fuzzy_lookup(pred["question_text"], crowd)
                crowd_p = hitc["crowd_pct"] / 100.0 if hitc else None
                sub = submission(fp, crowd_p, pred["question_family"], POSITION)
                sub = _shrink_no_edge(sub, pred["question_text"])
                sv = to_platform_probability(sub)
                if not (1 <= sv <= 99) or sv == 50:
                    continue                      # invariant guard
                if mk["id"] not in existing and mk["id"] not in journal:
                    new_payload.append({"market_id": mk["id"], "lobby_id": LOBBY_ID,
                                        "probability": sv, "_q": pred["question_text"]})
                    n_new += 1
                elif mk["id"] in journal and mk["id"] in existing:  # only OUR preds
                    try:
                        stored = int(round(float(existing[mk["id"]].get("probability"))))
                    except (TypeError, ValueError):
                        stored = -1
                    # closing window: track every change; earlier: only on a real
                    # line move (>= UPDATE_DELTA points) so we don't churn on noise
                    if stored != sv and (near or abs(sv - stored) >= UPDATE_DELTA):
                        updates.append((existing[mk["id"]]["id"], mk["id"], sv))
                        n_upd += 1
            logger.info("  %-18s T-%4.1fh round=%s  %d new  %d update",
                        m.get("name"), h, rnd, n_new, n_upd)
        except Exception as e:                   # noqa: BLE001 — per-match isolation
            s["errors"] += 1
            logger.warning("  %s: ERROR %s", m.get("name"), e)

    logger.info("  plan: %d to submit, %d to update", len(new_payload), len(updates))
    if not live:
        logger.info("=== %s: %d priced | would submit %d, update %d | %d skipped %d errors ===",
                    mode, s["priced"], len(new_payload), len(updates),
                    s["skipped"], s["errors"])
        return s

    # ---- live submission: per-batch isolation, per-result reconciliation ----
    for i in range(0, len(new_payload), 50):
        batch = new_payload[i:i + 50]
        payload = [{"market_id": p["market_id"], "lobby_id": p["lobby_id"],
                    "probability": p["probability"]} for p in batch]
        try:
            res = client.submit_batch(payload)
            results = res.get("results", []) if isinstance(res, dict) else []
            for r in results:
                mid = r.get("market_id")
                if r.get("success"):
                    s["submitted"] += 1
                    pid = (r.get("trade") or {}).get("id")
                    journal[mid] = {"prediction_id": pid, "submitted_at":
                                    datetime.now(timezone.utc).isoformat(timespec="seconds")}
                else:
                    logger.warning("    submit FAILED market=%s: %s", mid,
                                   r.get("error") or r)
            if not results:                       # unknown shape — assume all ok
                s["submitted"] += len(payload)
                for p in batch:
                    journal[p["market_id"]] = {"prediction_id": None}
        except Exception as e:                   # noqa: BLE001 — per-batch isolation
            logger.warning("    submit_batch error (%d markets) — will retry next "
                           "sweep: %s", len(payload), e)

    for pid, mid, sv in updates:
        try:
            client.update_prediction(pid, sv)
            s["updated"] += 1
            if mid in journal:
                journal[mid]["probability"] = sv
        except Exception as e:                   # noqa: BLE001
            logger.warning("    update %s failed: %s", mid, e)

    _save_journal(journal)
    logger.info("=== %s: %d priced | submitted %d | updated %d | %d skipped %d errors ===",
                mode, s["priced"], s["submitted"], s["updated"], s["skipped"], s["errors"])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="never submit; log only")
    args = ap.parse_args()
    _log_setup()
    if not _acquire_lock():
        return
    try:
        sweep(args.dry)
    except Exception as e:                        # noqa: BLE001
        logger.error("SWEEP FAILED: %s", e)
        sys.exit(1)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
