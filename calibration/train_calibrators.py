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
        ece_before = trainer.ece(probs, outs)
        cal = trainer.train(probs, outs)
        fixed = cal.predict(probs)
        ece_after = trainer.ece(fixed, outs)
        brier_before = float(np.mean((probs - outs) ** 2))
        brier_after = float(np.mean((fixed - outs) ** 2))
        print(f"{fam}: n={len(pp)}  ECE {ece_before:.4f} -> {ece_after:.4f}  "
              f"Brier {brier_before:.4f} -> {brier_after:.4f} (in-sample)")
        # ship only if the calibrator actually helps
        if brier_after <= brier_before + 1e-6:
            calibrators[fam] = cal
        else:
            print(f"  {fam}: calibrator made Brier worse — NOT shipped (identity).")

    CalibrationLayer(calibrators).save(args.out)
    print(f"Saved {len(calibrators)} calibrator(s) -> {args.out}")


if __name__ == "__main__":
    main()
