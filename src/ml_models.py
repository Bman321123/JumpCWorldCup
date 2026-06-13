"""Deployment wrappers for the gated ML micro-market models (ROADMAP C).

A model is loaded ONLY if it passed its walk-forward ship-gate at train time
(`ship_gate_pass`). If it failed, `active` is False and the orchestrator keeps
using the structural model — the gate is the whole point.

Feature vectors here must match ml/feature_store_micro.py exactly. At WC time
the per-team rates come from params; the structural probability comes from the
engine, so the GBM applies the same residual correction it learned on clubs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CornerMLModel:
    FEATURES = ["home_cf", "home_ca", "away_cf", "away_ca", "lam_struct",
                "threshold", "struct_prob"]

    def __init__(self, path: Optional[str] = None):
        self.active = False
        self.model = None
        self.calibrator = None
        self.gate = {}
        if path and Path(path).exists():
            try:
                import joblib
                blob = joblib.load(path)
                if blob.get("ship_gate_pass"):
                    self.model = blob["model"]
                    self.calibrator = blob["calibrator"]
                    self.gate = blob.get("walk_forward", {})
                    self.active = True
                    logger.info("Corner ML model active: %s", self.gate)
                else:
                    logger.info("Corner ML model present but FAILED gate — disabled.")
            except Exception as e:               # noqa: BLE001
                logger.warning("Corner ML load failed: %s", e)

    def prob_over(self, home_cf, home_ca, away_cf, away_ca, lam_struct,
                  threshold, struct_prob) -> Optional[float]:
        if not self.active:
            return None
        x = np.array([[home_cf, home_ca, away_cf, away_ca, lam_struct,
                       threshold, struct_prob]], dtype=float)
        p = float(self.model.predict_proba(x)[:, 1][0])
        p = float(self.calibrator.predict(np.array([p]))[0])
        return min(max(p, 0.01), 0.99)
