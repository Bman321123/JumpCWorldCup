"""Train + gate an ML model for every micro-market family (ROADMAP C+).

For each family: walk-forward by year, strict ship-gate (beat base rate AND the
structural model). Passers are saved active; failures are saved inactive so the
orchestrator keeps the structural model. Honest reporting per family.

  python ml/train_all.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# HistGradientBoosting's OpenMP parallelism can crash under nested backgrounding;
# cap threads for a stable nightly run.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_layer import CalibrationTrainer            # noqa: E402

# The devigged market line is NOT fed to the GBM — for goals the market is
# efficient and a partially-present feature destabilizes the learner. The
# line still combines with the model at the blend stage (§4.6), which is the
# right place for it.
FEATURES = ["home_for", "home_against", "away_for", "away_against",
            "home_goals", "away_goals", "lam_struct", "threshold", "struct_prob"]
MONO = {"threshold": -1, "struct_prob": 1, "lam_struct": 1}
FAMILIES = ["goals", "corners", "cards", "sot", "fouls"]


def make_model(features):
    mono = [MONO.get(f, 0) for f in features]
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=300,
                                  reg_lambda=1.0, monotone_constraints=mono,
                                  verbose=-1)
    except Exception:                            # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=300, l2_regularization=1.0, monotonic_cst=mono)


def train_family(fam: str) -> dict:
    path = ROOT / "data" / f"features_{fam}.csv"
    if not path.exists():
        return {"family": fam, "error": "no features"}
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    val_years = years[len(years) // 2:]
    # drop features that are entirely missing for this family (e.g. mkt on
    # non-goals) — a fully-NaN column breaks the monotone-constrained learner
    feats = [f for f in FEATURES if df[f].notna().any()]

    gbm_b, struct_b, base_b, n = [], [], [], 0
    for vy in val_years:
        tr = df[df["year"] < vy]
        va = df[df["year"] == vy]
        if len(tr) < 5000 or len(va) < 500:
            continue
        m = make_model(feats)
        m.fit(tr[feats], tr["label"])
        p = m.predict_proba(va[feats])[:, 1]
        base = tr.groupby("threshold")["label"].mean()
        bp = va["threshold"].map(base).to_numpy()
        y = va["label"].to_numpy()
        gbm_b.append(np.mean((p - y) ** 2))
        struct_b.append(np.mean((va["struct_prob"].clip(0, 1) - y) ** 2))
        base_b.append(np.mean((bp - y) ** 2))
        n += len(va)
    if not gbm_b:
        return {"family": fam, "error": "insufficient folds"}

    gbm, struct, base = float(np.mean(gbm_b)), float(np.mean(struct_b)), float(np.mean(base_b))
    passed = gbm < base - 1e-4 and gbm < struct - 1e-4

    cut = val_years[-1]
    tr = df[df["year"] < cut]; ho = df[df["year"] == cut]
    final = make_model(feats); final.fit(tr[feats], tr["label"])
    cal = CalibrationTrainer().train(final.predict_proba(ho[feats])[:, 1],
                                     ho["label"].to_numpy())
    final.fit(df[feats], df["label"])
    joblib.dump({"model": final, "calibrator": cal, "features": feats,
                 "ship_gate_pass": bool(passed),
                 "walk_forward": {"gbm": gbm, "structural": struct, "base": base}},
                ROOT / "params" / f"ml_{fam}.joblib")
    return {"family": fam, "n": n, "gbm": round(gbm, 5),
            "structural": round(struct, 5), "base": round(base, 5),
            "pass": passed}


def main() -> None:
    print(f"{'family':>8} {'n':>7} {'base':>8} {'struct':>8} {'gbm':>8}  gate")
    print("-" * 52)
    for fam in FAMILIES:
        try:
            r = train_family(fam)
        except Exception as e:                   # noqa: BLE001 — one family must not kill the rest
            print(f"{fam:>8}  CRASH: {e!r}")
            continue
        if "error" in r:
            print(f"{fam:>8}  {r['error']}")
            continue
        print(f"{fam:>8} {r['n']:>7} {r['base']:>8.5f} {r['structural']:>8.5f} "
              f"{r['gbm']:>8.5f}  {'PASS' if r['pass'] else 'fail'}")
    print("\nPassers ship active; failures stay inactive (structural model used).")


if __name__ == "__main__":
    main()
