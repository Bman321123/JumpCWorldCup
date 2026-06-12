"""Per-family probability calibration — model branch ONLY (PRD v2.2 §4.6, B5).

Trained on out-of-sample model predictions. Market-derived probabilities never
pass through these calibrators; sharp closing lines are already calibrated.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

ISOTONIC_MIN_N = 500
EPS = 1e-4


class PlattCalibrator:
    """Logistic regression on logit(p) — the small-sample fallback."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        x = _logit(np.clip(probs, EPS, 1 - EPS)).reshape(-1, 1)
        self.lr.fit(x, outcomes)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        x = _logit(np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


class CalibrationTrainer:
    def train(self, probs: np.ndarray, outcomes: np.ndarray):
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        if len(probs) >= ISOTONIC_MIN_N:
            ir = IsotonicRegression(out_of_bounds="clip", increasing=True,
                                    y_min=0.0, y_max=1.0)
            ir.fit(probs, outcomes)
            return ir
        logger.info("n=%d < %d: using Platt scaling.", len(probs), ISOTONIC_MIN_N)
        return PlattCalibrator().fit(probs, outcomes)

    @staticmethod
    def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        total = 0.0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            total += abs(outcomes[mask].mean() - probs[mask].mean()) * mask.sum()
        return float(total / len(probs))


class CalibrationLayer:
    def __init__(self, calibrators: Optional[Dict[str, object]] = None,
                 path: Optional[str] = None):
        if path is not None:
            calibrators = joblib.load(path)
        self.calibrators: Dict[str, object] = calibrators or {}

    def calibrate(self, raw_prob: float, family: str) -> float:
        cal = self.calibrators.get(family)
        if cal is None:
            return float(raw_prob)                      # identity fallback
        out = float(cal.predict(np.asarray([raw_prob]))[0])  # 1-D input (fixes B10)
        return float(np.clip(out, EPS, 1 - EPS))

    def save(self, path: str) -> None:
        joblib.dump(self.calibrators, path)
