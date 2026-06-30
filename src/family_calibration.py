"""Skill-weighted, SHRINK-ONLY calibration fit on our OWN settled results.

The contest is relative-Brier vs the crowd, so the winning posture is to express
confidence ONLY where we have demonstrably earned it. For each bet family this layer
measures how well our raw model number has tracked outcomes and pulls predictions
toward 50 in proportion to the EVIDENCE that we lack edge there.

Design choices that keep it from overfitting noise or stale data (the explicit
requirement):
  * SHRINK-ONLY. The per-family factor k is clamped to [k_min, 1]: we can dampen
    unearned confidence but NEVER amplify (no chasing a lucky run of overs). Worst
    case we converge to 50 = the crowd's hedge, which is safe on a relative scale.
  * Closed-form RIDGE toward k=1 ("the model is right"). k = (S_py+λ)/(S_pp+λ),
    where S_py=Σ(p-.5)(y-.5), S_pp=Σ(p-.5)². A family with little data or little
    signal is pulled to k≈1 (no change); only consistent, well-evidenced miscalibration
    moves k down. λ is the regularization strength (≈ pseudo-observations of "trust
    the model").
  * STRUCTURAL families, not teams/dates — corner-comparisons don't become
    predictable later in the tournament, so the factor generalizes forward; it is NOT
    a memorized table of past outcomes.
  * Shipped only if it beats raw out-of-sample (leave-one-out) — see fit()/main().
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

K_MIN = 0.35            # conservative floor: dampen overconfidence, don't nuke it
LAMBDA = 1.6           # strong ridge toward k=1; only well-evidenced families move,
                       # and they move GENTLY — low regret if the thin per-family
                       # sample is partly noise (the explicit anti-overfit requirement)
ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "params" / "family_calibration.json"


def family_key(question_text: str, question_family: Optional[str] = None) -> str:
    """Coarse, STRUCTURAL bet family. Uses the parser's question_family when given
    (clean player-vs-team split), else infers from the text."""
    q = question_text or ""
    ql = q.lower()
    fam = (question_family or "").upper()
    if "hydration" in ql or "cooling break" in ql:
        return "hydration"
    sot = "on target" in ql                           # matches "shot(s) on target"
    if "than" in ql:                                  # comparatives ("more X than Y")
        if sot:
            return "cmp_sot"
        if "corner" in ql:
            return "cmp_corners"
        if "foul" in ql or "card" in ql:
            return "cmp_fouls_cards"
        if "more total goals" in ql:
            return "half_h2_vs_h1"
    if fam == "PLAYER_MARKET" or "(" in q:
        return "player_sot" if sot else "player_goal"
    if "caught offside" in ql or "offside call" in ql or fam == "OFFSIDE_MARKET":
        return "offsides"
    if "penalt" in ql or "red card" in ql:
        return "penalty_red"
    if "first goal" in ql:
        return "first_goal"
    if sot:
        return "team_sot"
    if "corner" in ql:
        return "corners_total"
    if "card" in ql or fam == "CARD_MARKET":
        return "cards_total"
    if "win" in ql or "ahead at halftime" in ql or ("halftime" in ql and "tied" in ql):
        return "result"
    if "both teams score" in ql:
        return "btts"
    if "score" in ql and ("half" in ql or "at least 1 goal" in ql):
        return "team_score_window"
    if "total goals" in ql or fam == "GOAL_MARKET":
        return "goals_total"
    return "other"


def _factor(rows: List[Tuple[float, float]], lam: float = LAMBDA) -> float:
    """Closed-form ridge shrink toward 50, clamped shrink-only to [K_MIN, 1]."""
    s_py = sum((p - 0.5) * (y - 0.5) for p, y in rows)
    s_pp = sum((p - 0.5) ** 2 for p, y in rows)
    k = (s_py + lam) / (s_pp + lam)
    return max(K_MIN, min(1.0, k))


def fit(rows: List[Tuple[float, float, str]], min_n: int = 8) -> Dict[str, dict]:
    """rows = [(p_model, y, family)]. Returns {family: {k, n, brier_raw, brier_cal}}.
    Families below min_n keep k=1 (not enough evidence to adjust)."""
    by: Dict[str, List[Tuple[float, float]]] = {}
    for p, y, f in rows:
        by.setdefault(f, []).append((p, y))
    out = {}
    for f, rs in by.items():
        n = len(rs)
        k = _factor(rs) if n >= min_n else 1.0
        braw = sum((p - y) ** 2 for p, y in rs) / n
        bcal = sum((0.5 + (p - 0.5) * k - y) ** 2 for p, y in rs) / n
        out[f] = {"k": round(k, 4), "n": n,
                  "brier_raw": round(braw, 4), "brier_cal": round(bcal, 4)}
    return out


def apply(p: float, question_text: str, question_family: Optional[str],
          table: Optional[Dict[str, dict]]) -> float:
    """Pull p toward 0.5 by the family's earned-confidence factor k (k=1 if unknown)."""
    if not table:
        return p
    rec = table.get(family_key(question_text, question_family))
    k = rec["k"] if rec else 1.0
    return 0.5 + (p - 0.5) * k


def load_table(path: Path = TABLE_PATH) -> Optional[Dict[str, dict]]:
    try:
        return json.loads(path.read_text()).get("families")
    except (FileNotFoundError, ValueError):
        return None


def loo_brier(rows: List[Tuple[float, float, str]], min_n: int = 8
              ) -> Tuple[float, float]:
    """Honest out-of-sample check: for each row, recompute its family's factor with
    that row HELD OUT, then score it. Returns (raw_brier, calibrated_brier)."""
    by: Dict[str, List[Tuple[float, float]]] = {}
    for p, y, f in rows:
        by.setdefault(f, []).append((p, y))
    raw = sum((p - y) ** 2 for p, y, _ in rows) / len(rows)
    tot = 0.0
    for p, y, f in rows:
        rest = [(pp, yy) for pp, yy in by[f] if not (pp == p and yy == y)]
        # remove exactly one matching instance
        seen = False
        rest = []
        for pp, yy in by[f]:
            if not seen and pp == p and yy == y:
                seen = True
                continue
            rest.append((pp, yy))
        k = _factor(rest) if len(rest) >= min_n - 1 and len(by[f]) >= min_n else 1.0
        tot += (0.5 + (p - 0.5) * k - y) ** 2
    return raw, tot / len(rows)


def _load_rows(db: str, since: Optional[str] = None) -> List[Tuple[float, float, str]]:
    import sqlite3
    con = sqlite3.connect(db)
    # `since` restricts to the CURRENT model regime — fitting across a code change
    # (e.g. the landmine fixes) would teach the layer to undo bugs already fixed.
    where = ("WHERE brier_contribution IS NOT NULL AND p_blend IS NOT NULL"
             + (" AND substr(submitted_at,1,10) >= ?" if since else ""))
    args = (since,) if since else ()
    rows = []
    for qt, qf, p_blend, sub, brier in con.execute(
            "SELECT question_text, question_family, p_blend, submitted_probability, "
            f"brier_contribution FROM predictions_log {where}", args):
        # Drop rows corrupted by now-FIXED bugs so we don't calibrate to undo them:
        # mojibake markers ("SuÄiÄ") flag the accented-name era where player props were
        # misrouted to team markets at ~0.99. Those questions price correctly now, so
        # their stale p_blend must not train the calibrator (the explicit anti-overfit
        # requirement). Post-fix data has no mojibake, so this only prunes the past.
        if any(m in (qt or "") for m in ("Ã", "Ä", "Å", "Â")):
            continue
        ps = float(sub) / 100.0 if sub is not None else float(p_blend)
        y = 1.0 if abs((ps - 1.0) ** 2 - float(brier)) < 1e-4 else 0.0
        rows.append((float(p_blend), y, family_key(qt, qf)))
    con.close()
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--out", default=str(TABLE_PATH))
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--since", default=None,
                    help="only fit on results from this date on (current model regime)")
    ap.add_argument("--write", action="store_true",
                    help="write the table only if it beats raw out-of-sample")
    args = ap.parse_args()
    rows = _load_rows(args.db, args.since)
    fams = fit(rows, args.min_n)
    raw, cal = loo_brier(rows, args.min_n)
    print(f"fit on {len(rows)} settled rows | LOO Brier raw {raw:.4f} -> calibrated "
          f"{cal:.4f}  ({raw - cal:+.4f})")
    print(f"{'family':18} {'n':>3} {'k':>5} {'brier_raw':>9} {'brier_cal':>9}")
    for f, r in sorted(fams.items(), key=lambda kv: kv[1]["k"]):
        flag = "  <- shrink" if r["k"] < 0.97 else ""
        print(f"  {f:16} {r['n']:>3} {r['k']:>5.2f} {r['brier_raw']:>9.4f} "
              f"{r['brier_cal']:>9.4f}{flag}")
    if args.write:
        if cal < raw - 1e-4:
            import datetime  # noqa: F401 — only for a static stamp passed externally
            Path(args.out).write_text(json.dumps(
                {"_note": "skill-weighted shrink-only calibration; refit nightly",
                 "loo_raw": round(raw, 4), "loo_cal": round(cal, 4),
                 "families": fams}, indent=1))
            print(f"WROTE {args.out} (improves out-of-sample)")
        else:
            print("NOT written — does not beat raw out-of-sample (no overfit shipped)")


if __name__ == "__main__":
    main()
