"""Corner-market feature store from the club corpus (ROADMAP C).

Features are RATE-based, not raw rolling sums, so the trained model deploys
unchanged on World Cup matches (where the rates come from params instead of a
rolling window). Each row also carries the structural Poisson probability the
engine would output, so the GBM learns a RESIDUAL correction and the ship-gate
can check it actually beats the structural model.

  python ml/feature_store_micro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
CLUB = ROOT / "data" / "club_matches.csv"
OUT = ROOT / "data" / "features_corners.csv"

WINDOW = 20
MIN_HISTORY = 8
THRESHOLDS = [8.5, 9.5, 10.5, 11.5]


def rolling_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team corner for/against, rolling mean of the prior WINDOW matches."""
    rows = []
    for _, m in df.iterrows():
        rows.append({"team": m["home"], "date": m["date"],
                     "cf": m["hc"], "ca": m["ac"], "mid": m.name})
        rows.append({"team": m["away"], "date": m["date"],
                     "cf": m["ac"], "ca": m["hc"], "mid": m.name})
    tl = pd.DataFrame(rows).sort_values("date")
    g = tl.groupby("team")
    tl["cf_roll"] = g["cf"].transform(
        lambda s: s.shift().rolling(WINDOW, min_periods=MIN_HISTORY).mean())
    tl["ca_roll"] = g["ca"].transform(
        lambda s: s.shift().rolling(WINDOW, min_periods=MIN_HISTORY).mean())
    tl["n_prior"] = g.cumcount()
    return tl


def build() -> pd.DataFrame:
    df = pd.read_csv(CLUB, parse_dates=["date"]).reset_index(drop=True)
    df = df.dropna(subset=["hc", "ac"]).sort_values("date").reset_index()
    df = df.rename(columns={"index": "orig"})
    df.index.name = None
    tl = rolling_rates(df)

    # split timeline back into home/away rate lookups by match id
    home_tl = tl.iloc[0::2].set_index("mid")
    away_tl = tl.iloc[1::2].set_index("mid")
    league_avg = (df["hc"] + df["ac"]).mean() / 2.0

    rows = []
    for mid, m in df.iterrows():
        h = home_tl.loc[mid] if mid in home_tl.index else None
        a = away_tl.loc[mid] if mid in away_tl.index else None
        if h is None or a is None:
            continue
        if not np.isfinite(h["cf_roll"]) or not np.isfinite(a["cf_roll"]):
            continue
        hcf, hca, acf, aca = h["cf_roll"], h["ca_roll"], a["cf_roll"], a["ca_roll"]
        lam_h = hcf * (aca / league_avg)
        lam_a = acf * (hca / league_avg)
        lam = lam_h + lam_a
        total = m["hc"] + m["ac"]
        for thr in THRESHOLDS:
            k = int(np.ceil(thr))
            struct = float(1.0 - poisson.cdf(k - 1, lam))
            rows.append({"date": m["date"], "home_cf": hcf, "home_ca": hca,
                         "away_cf": acf, "away_ca": aca,
                         "lam_struct": lam, "threshold": thr,
                         "struct_prob": struct,
                         "label": int(total >= thr)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"corner features: {len(out)} rows from {len(df)} matches -> {OUT}")
    return out


if __name__ == "__main__":
    if not CLUB.exists():
        sys.exit("Run ml/ingest_club.py first.")
    build()
