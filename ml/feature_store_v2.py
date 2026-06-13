"""Unified micro-market feature store across all count families (ROADMAP C+).

One rolling-rate pass over the club corpus produces deployment-ready, rate-based
feature rows for every family (goals/corners/cards/SOT/fouls). Each row carries
the structural Poisson probability (so the GBM learns a residual and the gate
can check it beats structural) and, for goals, the devigged market total prob
(so the GBM can learn where the line is slightly off — the line+ML combination).

  python ml/feature_store_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
CLUB = ROOT / "data" / "club_matches.csv"
WINDOW = 20
MIN_HISTORY = 8

# family -> (for/against source per home perspective, thresholds, opp_adjust)
FAMILIES = {
    "goals":   {"thr": [1.5, 2.5, 3.5], "opp": True},
    "corners": {"thr": [8.5, 9.5, 10.5, 11.5], "opp": True},
    "cards":   {"thr": [2.5, 3.5, 4.5, 5.5], "opp": False},
    "sot":     {"thr": [6.5, 7.5, 8.5, 9.5], "opp": True},
    "fouls":   {"thr": [18.5, 21.5, 24.5, 27.5], "opp": False},
}


def _devig_total(o, u):
    if not (o and u) or o <= 1 or u <= 1:
        return np.nan
    io, iu = 1.0 / o, 1.0 / u
    return io / (io + iu)


def build() -> None:
    df = pd.read_csv(CLUB, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["mid"] = df.index
    # per-team stat values for the home and away perspectives
    df["g_h"], df["g_a"] = df["fthg"], df["ftag"]
    df["c_h"], df["c_a"] = df["hc"], df["ac"]
    df["k_h"] = df["hy"].fillna(0) + df["hr"].fillna(0)
    df["k_a"] = df["ay"].fillna(0) + df["ar"].fillna(0)
    df["s_h"], df["s_a"] = df["hst"], df["ast"]
    df["f_h"], df["f_a"] = df["hf"], df["af"]
    metrics = {"goals": ("g_h", "g_a"), "corners": ("c_h", "c_a"),
               "cards": ("k_h", "k_a"), "sot": ("s_h", "s_a"),
               "fouls": ("f_h", "f_a")}

    # build a team timeline (home + away perspective rows)
    persp = []
    for side in ("home", "away"):
        sub = pd.DataFrame({"mid": df["mid"], "date": df["date"],
                            "team": df["home"] if side == "home" else df["away"]})
        for fam, (hcol, acol) in metrics.items():
            if side == "home":
                sub[f"{fam}_for"] = df[hcol]
                sub[f"{fam}_against"] = df[acol]
            else:
                sub[f"{fam}_for"] = df[acol]
                sub[f"{fam}_against"] = df[hcol]
        sub["side"] = side
        persp.append(sub)
    tl = pd.concat(persp).sort_values("date")
    roll_cols = [c for c in tl.columns if c.endswith("_for") or c.endswith("_against")]
    g = tl.groupby("team")
    for c in roll_cols:
        tl[c + "_r"] = g[c].transform(
            lambda s: s.shift().rolling(WINDOW, min_periods=MIN_HISTORY).mean())
    home = tl[tl.side == "home"].set_index("mid")
    away = tl[tl.side == "away"].set_index("mid")

    league_avg = {fam: (df[hc] + df[ac]).mean() / 2.0
                  for fam, (hc, ac) in metrics.items()}
    df["mkt_goals25"] = [_devig_total(o, u) for o, u in zip(df["o25o"], df["o25u"])] \
        if "o25o" in df.columns else np.nan

    for fam, cfg in FAMILIES.items():
        hc, ac = metrics[fam]
        rows = []
        for mid in df.index:
            if mid not in home.index or mid not in away.index:
                continue
            hf = home.at[mid, f"{fam}_for_r"]; hg = home.at[mid, f"{fam}_against_r"]
            af = away.at[mid, f"{fam}_for_r"]; ag = away.at[mid, f"{fam}_against_r"]
            if not np.isfinite(hf) or not np.isfinite(af):
                continue
            avg = league_avg[fam]
            if cfg["opp"]:
                lam = hf * (ag / avg) + af * (hg / avg)
            else:
                lam = hf + af
            total = df.at[mid, hc] + df.at[mid, ac]
            # general strength context
            gh = home.at[mid, "goals_for_r"]; ga = away.at[mid, "goals_for_r"]
            mkt = df.at[mid, "mkt_goals25"] if fam == "goals" else np.nan
            for thr in cfg["thr"]:
                k = int(np.ceil(thr))
                struct = float(1.0 - poisson.cdf(k - 1, lam))
                rows.append({"date": df.at[mid, "date"], "home_for": hf,
                             "home_against": hg, "away_for": af, "away_against": ag,
                             "home_goals": gh, "away_goals": ga, "lam_struct": lam,
                             "threshold": thr, "struct_prob": struct,
                             "mkt": mkt if (fam == "goals" and abs(thr - 2.5) < 0.01)
                                    else np.nan,
                             "label": int(total >= thr)})
        out = pd.DataFrame(rows)
        path = ROOT / "data" / f"features_{fam}.csv"
        out.to_csv(path, index=False)
        print(f"  {fam}: {len(out)} rows -> {path.name}")


if __name__ == "__main__":
    if not CLUB.exists():
        sys.exit("Run ml/ingest_club.py first.")
    print("building unified feature store…")
    build()
