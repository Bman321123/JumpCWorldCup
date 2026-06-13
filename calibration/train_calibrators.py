"""Train per-family calibrators from as-of backtest replays (PRD §4.6, B5).

Collects (model probability, outcome) pairs across tournaments — every fit is
strictly as-of, so the pairs are honest out-of-sample model behavior — and
fits the model-branch calibrators. Market probabilities are never calibrated.

  python calibration/train_calibrators.py
  python calibration/train_calibrators.py --tournaments WC2018,WC2022,EURO2024,COPA2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.replay import TOURNAMENTS, replay              # noqa: E402
from src.calibration_layer import (CalibrationLayer,         # noqa: E402
                                   CalibrationTrainer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--tournaments", default="WC2018,WC2022,EURO2024,COPA2024")
    ap.add_argument("--out", default=str(ROOT / "params" / "calibrators.joblib"))
    args = ap.parse_args()

    pairs: dict = {}
    for name in args.tournaments.split(","):
        name = name.strip()
        if name not in TOURNAMENTS:
            continue
        print(f"Replaying {name} (as-of fit)…")
        rep = replay(args.db, name, TOURNAMENTS[name])
        if "error" in rep:
            print(" ", rep["error"])
            continue
        for fam, pp in rep["pairs"].items():
            pairs.setdefault(fam, []).extend(pp)

    trainer = CalibrationTrainer()
    calibrators = {}
    for fam, pp in pairs.items():
        probs = np.array([p for p, _ in pp])
        outs = np.array([o for _, o in pp])

        # ship gate: 5-fold CV — the calibrator must beat raw Brier on folds
        # it never saw (in-sample ECE alone is how the first version shipped a
        # calibrator that mangled out-of-band inputs)
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(probs))
        cv_raw, cv_cal = [], []
        for fold in np.array_split(idx, 5):
            mask = np.ones(len(probs), bool)
            mask[fold] = False
            cal_f = trainer.train(probs[mask], outs[mask])
            pred_f = cal_f.predict(probs[fold])
            cv_raw.append(np.mean((probs[fold] - outs[fold]) ** 2))
            cv_cal.append(np.mean((pred_f - outs[fold]) ** 2))
        brier_raw, brier_cal = float(np.mean(cv_raw)), float(np.mean(cv_cal))
        print(f"{fam}: n={len(pp)}  CV Brier raw {brier_raw:.4f} vs "
              f"calibrated {brier_cal:.4f}")
        if brier_cal <= brier_raw + 1e-6:
            cal = trainer.train(probs, outs)
            calibrators[fam] = cal
            print(f"  shipped (guard range [{cal.lo:.2f}, {cal.hi:.2f}]; "
                  "identity outside)")
        else:
            print(f"  {fam}: NOT shipped — raw model wins on held-out folds.")

    CalibrationLayer(calibrators).save(args.out)
    print(f"Saved {len(calibrators)} calibrator(s) -> {args.out}")


if __name__ == "__main__":
    main()
