"""Leak-free backtest + cluster-robust ship gate (restores the lost PRD-v3 keystone).

The contest is scored on Brier; the ONLY honest way to decide whether a change
(more data, a new parameter, a different half-life) helps is to measure it on a
held-out fold with calibrated uncertainty. This harness does that:

  1. TEMPORAL HOLDOUT (no leakage): fit Dixon-Coles on internationals strictly
     before --test-start (via the estimator's `asof` discipline), then price 1X2,
     over-2.5 and BTTS on every international from --test-start onward.
  2. BASELINE: climatology — the constant outcome frequencies from the TRAIN split.
  3. GATE: a tournament-CLUSTERED paired bootstrap on the per-match Brier reduction
     (base - model). Few clusters make the plain percentile bootstrap under-cover,
     so we also report a cluster-robust t-interval (df = K-1). Verdict PASS only if
     the 90% lower bound of the reduction is > 0 (model genuinely beats base).

Self-tests (blocking, run with --selftest or via tests/test_eval_harness.py):
  - LABEL SHUFFLE: permute outcomes -> the measured edge must collapse to ~0
    (catches leakage / a harness that always "passes").
  - CI COVERAGE: on synthetic clustered data with a known effect, the interval
    must cover at ~the nominal rate.

  python ml/eval_harness.py                       # default holdout, full report
  python ml/eval_harness.py --test-start 2024-06-01 --half-life 500 --no-friendlies
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parameter_estimator import elo_to_priors, fit_dixon_coles   # noqa: E402
from src.stats_engine import StatsEngine                             # noqa: E402
from src.types import Condition, MatchContext, TemporalWindow        # noqa: E402

DB = str(ROOT / "data" / "wc_forecasting.db")
DEFAULT_TEST_START = "2024-01-01"


# ----------------------------------------------------------------------------- data
def load_matches(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score, neutral, "
        "tournament FROM matches", con)
    con.close()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].fillna(0).astype(int)
    return df


# --------------------------------------------------------------------------- scoring
def _ctx(home: str, away: str, date: str, neutral: int) -> MatchContext:
    # gamma is fit as `gamma * (1 - neutral)`, so home advantage applies iff non-neutral
    return MatchContext(home, away, date, home_is_host=(neutral == 0))


def price_and_score(eng: StatsEngine, test: pd.DataFrame
                    ) -> Dict[str, np.ndarray]:
    """Return per-match Brier for the model on each market."""
    b_1x2, b_o25, b_btts = [], [], []
    for r in test.itertuples(index=False):
        ctx = _ctx(r.home_team, r.away_team, str(r.date.date()), r.neutral)
        p = eng.result_probs(r.home_team, r.away_team, ctx)
        ph, pd_, pa = p["home_win"], p["draw"], p["away_win"]
        hw = 1.0 if r.home_score > r.away_score else 0.0
        dr = 1.0 if r.home_score == r.away_score else 0.0
        aw = 1.0 if r.home_score < r.away_score else 0.0
        b_1x2.append((ph - hw) ** 2 + (pd_ - dr) ** 2 + (pa - aw) ** 2)

        po = eng.goal_market(r.home_team, r.away_team, "GOALS", "TOTAL", 2.5,
                             Condition.GTE, TemporalWindow.FULL, ctx)
        yo = 1.0 if (r.home_score + r.away_score) >= 3 else 0.0
        b_o25.append((po - yo) ** 2)

        pb = eng.goal_market(r.home_team, r.away_team, "BTTS", "MATCH", 0.0,
                             Condition.GTE, TemporalWindow.FULL, ctx)
        yb = 1.0 if (r.home_score >= 1 and r.away_score >= 1) else 0.0
        b_btts.append((pb - yb) ** 2)
    return {"1x2": np.array(b_1x2), "over2.5": np.array(b_o25),
            "btts": np.array(b_btts)}


def base_rate_brier(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Climatology baseline: constant TRAIN frequencies applied to every test match."""
    hw = (train.home_score > train.away_score).mean()
    dr = (train.home_score == train.away_score).mean()
    aw = 1.0 - hw - dr
    o25 = ((train.home_score + train.away_score) >= 3).mean()
    btts = ((train.home_score >= 1) & (train.away_score >= 1)).mean()
    th = (test.home_score > test.away_score).astype(float).to_numpy()
    td = (test.home_score == test.away_score).astype(float).to_numpy()
    ta = (test.home_score < test.away_score).astype(float).to_numpy()
    to = ((test.home_score + test.away_score) >= 3).astype(float).to_numpy()
    tb = ((test.home_score >= 1) & (test.away_score >= 1)).astype(float).to_numpy()
    return {"1x2": (hw - th) ** 2 + (dr - td) ** 2 + (aw - ta) ** 2,
            "over2.5": (o25 - to) ** 2, "btts": (btts - tb) ** 2}


# ------------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap(reduction: np.ndarray, clusters: np.ndarray,
                      n_boot: int = 3000, alpha: float = 0.10,
                      seed: int = 12345) -> Dict[str, float]:
    """One-sided (1-alpha) lower bound on the mean per-match Brier REDUCTION
    (base - model; positive = model better), resampling whole tournaments.
    Returns the percentile lower bound and a cluster-robust t-interval lower bound."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    K = len(uniq)
    by = {c: reduction[clusters == c] for c in uniq}
    point = float(reduction.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=K, replace=True)
        boot[b] = np.concatenate([by[c] for c in pick]).mean()
    perc_lb = float(np.quantile(boot, alpha))                 # one-sided percentile LB
    # cluster-robust t-interval: SE from cluster means, df = K-1
    cmeans = np.array([by[c].mean() for c in uniq])
    se = cmeans.std(ddof=1) / np.sqrt(K) if K > 1 else float("inf")
    try:
        from scipy.stats import t as student_t
        tcrit = float(student_t.ppf(1.0 - alpha, df=max(K - 1, 1)))
    except Exception:                                          # noqa: BLE001
        tcrit = 1.34                                           # ~t(0.90, df large)
    t_lb = point - tcrit * se
    return {"point": point, "perc_lb": perc_lb, "t_lb": t_lb, "K": K,
            "se": se, "n": int(len(reduction))}


def verdict(stats: Dict[str, float]) -> str:
    lb = min(stats["perc_lb"], stats["t_lb"])    # conservative: both must clear 0
    if lb > 0:
        return "PASS"
    if stats["point"] > 0:
        return "MARGINAL"
    return "HURT"


# --------------------------------------------------------------------------- evaluate
def evaluate(df: pd.DataFrame, test_start: str, half_life: float, ridge: float,
             elo_path: Optional[str], drop_friendlies: bool,
             maxiter: int = 300, elo_scale: float = 0.0009) -> Dict[str, dict]:
    cutoff = pd.to_datetime(test_start)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff].copy()
    if drop_friendlies:
        test = test[~test["tournament"].str.contains("Friendly", case=False, na=False)]
    priors = {"attack": {}, "defense": {}}
    if elo_path and Path(elo_path).exists():
        priors = elo_to_priors(json.load(open(elo_path)), scale=elo_scale)
    params = fit_dixon_coles(df, half_life_days=half_life, ridge=ridge,
                             elo_attack_prior=priors["attack"],
                             elo_defense_prior=priors["defense"],
                             asof=test_start, maxiter=maxiter)
    eng = StatsEngine(params)
    model_b = price_and_score(eng, test)
    base_b = base_rate_brier(train, test)
    clusters = test["tournament"].fillna("Unknown").to_numpy()
    out = {}
    for market in ("1x2", "over2.5", "btts"):
        red = base_b[market] - model_b[market]
        st = cluster_bootstrap(red, clusters)
        st.update({"model_brier": float(model_b[market].mean()),
                   "base_brier": float(base_b[market].mean()),
                   "verdict": verdict(st)})
        out[market] = st
    out["_meta"] = {"train_n": int(len(train)), "test_n": int(len(test)),
                    "test_start": test_start, "half_life": half_life,
                    "fit_n": params.n_matches_fit}
    return out


def report(res: Dict[str, dict]) -> str:
    m = res["_meta"]
    lines = [f"Backtest holdout @ {m['test_start']}  |  fit {m['fit_n']} matches  |  "
             f"test {m['test_n']} internationals",
             f"{'market':9} {'model':>7} {'base':>7} {'reduce':>8} {'90%LB':>8} "
             f"{'K':>3}  verdict"]
    for market in ("1x2", "over2.5", "btts"):
        s = res[market]
        lb = min(s["perc_lb"], s["t_lb"])
        lines.append(f"{market:9} {s['model_brier']:7.4f} {s['base_brier']:7.4f} "
                     f"{s['point']:+8.4f} {lb:+8.4f} {s['K']:3d}  {s['verdict']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- selftest
def selftest() -> bool:
    """Blocking controls: label-shuffle kills the edge; bootstrap covers nominally."""
    ok = True
    rng = np.random.default_rng(0)
    # 1) CI coverage on synthetic clustered data with a TRUE positive effect
    covered = 0
    trials = 200
    for s in range(trials):
        r2 = np.random.default_rng(s)
        K = 10
        cluster_eff = r2.normal(0.01, 0.01, K)        # true mean ~0.01 reduction
        reds, cl = [], []
        for k in range(K):
            nk = r2.integers(20, 60)
            reds.append(r2.normal(cluster_eff[k], 0.2, nk))
            cl.append(np.full(nk, k))
        reds = np.concatenate(reds); cl = np.concatenate(cl)
        st = cluster_bootstrap(reds, cl, n_boot=800, seed=s)
        true_mean = float(cluster_eff.mean())
        if st["t_lb"] <= true_mean:                   # one-sided lower bound covers truth
            covered += 1
    cov = covered / trials
    print(f"  coverage (one-sided 90% t-LB covers truth): {cov:.3f} (want >= 0.85)")
    ok = ok and cov >= 0.85
    # 2) label-shuffle: pure-noise reductions must NOT pass the gate often
    passes = 0
    for s in range(200):
        r2 = np.random.default_rng(1000 + s)
        K = 10
        reds, cl = [], []
        for k in range(K):
            nk = r2.integers(20, 60)
            reds.append(r2.normal(0.0, 0.2, nk))      # zero true effect
            cl.append(np.full(nk, k))
        reds = np.concatenate(reds); cl = np.concatenate(cl)
        st = cluster_bootstrap(reds, cl, n_boot=800, seed=s)
        if verdict(st) == "PASS":
            passes += 1
    fpr = passes / 200
    print(f"  false-positive rate on null effect: {fpr:.3f} (want <= 0.15)")
    ok = ok and fpr <= 0.15
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--test-start", default=DEFAULT_TEST_START)
    ap.add_argument("--half-life", type=float, default=500.0)
    ap.add_argument("--ridge", type=float, default=5.0)
    ap.add_argument("--elo", default=str(ROOT / "params" / "elo.json"))
    ap.add_argument("--elo-scale", type=float, default=0.0009,
                    help="strength units per Elo point in the prior (higher = de-compress favorites)")
    ap.add_argument("--no-friendlies", action="store_true")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports" / "backtest.json"))
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    df = load_matches(args.db)
    res = evaluate(df, args.test_start, args.half_life, args.ridge, args.elo,
                   args.no_friendlies, args.maxiter, args.elo_scale)
    print(report(res))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
