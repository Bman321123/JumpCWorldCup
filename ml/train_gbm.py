"""GBM training with walk-forward validation and ship-gates (PRD v2.2 §8.2-8.4).

- LightGBM if available, else sklearn HistGradientBoosting (both support the
  monotone constraint: P(over) non-increasing in threshold).
- Walk-forward by year — NEVER random K-fold.
- Ship gate: walk-forward Brier must beat the per-question-type base rate.
- Output: ml_goal_model.joblib {model, calibrator, features, report}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_layer import CalibrationTrainer  # noqa: E402

VALID_YEARS = (2022, 2023, 2024, 2025)
EXCLUDE = {"kickoff", "home", "away", "label"}


def make_model(feature_names):
    mono = [-1 if f == "threshold" else 0 for f in feature_names]
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, reg_alpha=0.1, reg_lambda=1.0,
            monotone_constraints=mono, verbose=-1)
    except Exception:        # noqa: BLE001 — e.g. missing libomp on macOS
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, monotonic_cst=mono)


def walk_forward(df: pd.DataFrame, feature_names) -> dict:
    report = {}
    for year in VALID_YEARS:
        train = df[df["kickoff"].dt.year < year]
        valid = df[df["kickoff"].dt.year == year]
        if len(valid) < 50 or len(train) < 2000:
            continue
        model = make_model(feature_names)
        model.fit(train[feature_names], train["label"])
        p = model.predict_proba(valid[feature_names])[:, 1]
        brier = float(np.mean((p - valid["label"]) ** 2))
        # base-rate baseline per question type from the training window only
        base = train.groupby(["metric_btts", "threshold"])["label"].mean()
        bp = valid.apply(lambda r: base.get((r["metric_btts"], r["threshold"]), 0.5),
                         axis=1).to_numpy()
        brier_base = float(np.mean((bp - valid["label"]) ** 2))
        report[year] = {"n": int(len(valid)), "brier_gbm": round(brier, 5),
                        "brier_base_rate": round(brier_base, 5),
                        "gate_pass": bool(brier < brier_base)}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "features_goal.csv"))
    ap.add_argument("--out", default=str(ROOT / "params" / "ml_goal_model.joblib"))
    args = ap.parse_args()

    df = pd.read_csv(args.features, parse_dates=["kickoff"])
    feature_names = [c for c in df.columns if c not in EXCLUDE]
    print(f"{len(df)} rows, {len(feature_names)} features")

    report = walk_forward(df, feature_names)
    print(f"{'year':>6} {'n':>7} {'gbm':>9} {'base':>9}  gate")
    for year, r in report.items():
        print(f"{year:>6} {r['n']:>7} {r['brier_gbm']:>9} {r['brier_base_rate']:>9}  "
              f"{'PASS' if r['gate_pass'] else 'FAIL'}")
    gate = all(r["gate_pass"] for r in report.values()) and len(report) > 0
    print(f"SHIP GATE: {'PASS' if gate else 'FAIL — family stays on structural logic'}")

    # final model on all data + isotonic calibrator on out-of-fold-style holdout
    holdout = df[df["kickoff"].dt.year >= VALID_YEARS[0]]
    train = df[df["kickoff"].dt.year < VALID_YEARS[0]]
    model = make_model(feature_names)
    model.fit(train[feature_names], train["label"])
    p_hold = model.predict_proba(holdout[feature_names])[:, 1]
    calibrator = CalibrationTrainer().train(p_hold, holdout["label"].to_numpy())
    ece = CalibrationTrainer.ece(p_hold, holdout["label"].to_numpy())
    print(f"holdout ECE (pre-calibration): {ece:.4f}")

    final = make_model(feature_names)
    final.fit(df[feature_names], df["label"])
    joblib.dump({"model": final, "calibrator": calibrator,
                 "features": feature_names, "report": report,
                 "ship_gate_pass": gate}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
