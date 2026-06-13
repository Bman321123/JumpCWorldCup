"""Train + gate the corner-market ML model (ROADMAP C).

Walk-forward by year. Ship gate is strict: the GBM must beat BOTH the base rate
AND the structural Poisson model (which is one of its own input features) on
held-out folds. If it can't, it does not ship and corners stay structural.

  python ml/train_micro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_layer import CalibrationTrainer            # noqa: E402

FEATURES = ["home_cf", "home_ca", "away_cf", "away_ca", "lam_struct",
            "threshold", "struct_prob"]
MONO = {"threshold": -1, "struct_prob": 1, "lam_struct": 1}


def make_model():
    mono = [MONO.get(f, 0) for f in FEATURES]
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=200,
                                  reg_lambda=1.0, monotone_constraints=mono,
                                  verbose=-1)
    except Exception:                            # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0, monotonic_cst=mono)


def main() -> None:
    feats = ROOT / "data" / "features_corners.csv"
    if not feats.exists():
        sys.exit("Run ml/feature_store_micro.py first.")
    df = pd.read_csv(feats, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    val_years = years[len(years) // 2:]          # back half walks forward

    gbm_b, struct_b, base_b, n = [], [], [], 0
    for vy in val_years:
        train = df[df["year"] < vy]
        valid = df[df["year"] == vy]
        if len(train) < 3000 or len(valid) < 300:
            continue
        model = make_model()
        model.fit(train[FEATURES], train["label"])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        base = train.groupby("threshold")["label"].mean()
        bp = valid["threshold"].map(base).to_numpy()
        y = valid["label"].to_numpy()
        gbm_b.append(np.mean((p - y) ** 2))
        struct_b.append(np.mean((valid["struct_prob"] - y) ** 2))
        base_b.append(np.mean((bp - y) ** 2))
        n += len(valid)

    gbm, struct, base = np.mean(gbm_b), np.mean(struct_b), np.mean(base_b)
    print(f"walk-forward Brier (n={n}):")
    print(f"  base rate : {base:.5f}")
    print(f"  structural: {struct:.5f}")
    print(f"  GBM       : {gbm:.5f}")
    beats_struct = gbm < struct - 1e-4
    beats_base = gbm < base - 1e-4
    passed = beats_struct and beats_base
    print(f"  beats structural: {beats_struct} | beats base: {beats_base}")
    print(f"SHIP GATE: {'PASS' if passed else 'FAIL — corners stay structural'}")

    # fit final model on all but the last year, calibrate on the last year
    cut = val_years[-1]
    train = df[df["year"] < cut]
    hold = df[df["year"] == cut]
    final = make_model()
    final.fit(train[FEATURES], train["label"])
    p_hold = final.predict_proba(hold[FEATURES])[:, 1]
    calibrator = CalibrationTrainer().train(p_hold, hold["label"].to_numpy())
    final.fit(df[FEATURES], df["label"])         # refit on everything
    out = ROOT / "params" / "ml_corners.joblib"
    joblib.dump({"model": final, "calibrator": calibrator, "features": FEATURES,
                 "ship_gate_pass": bool(passed),
                 "walk_forward": {"gbm": float(gbm), "structural": float(struct),
                                  "base": float(base)}}, out)
    print(f"Saved {out} (ship_gate_pass={passed})")


if __name__ == "__main__":
    main()
