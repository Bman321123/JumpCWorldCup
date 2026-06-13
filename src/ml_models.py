"""Deployment registry for the gated ML micro-market models (ROADMAP C+).

A family's model is ACTIVE only if it passed its walk-forward ship-gate
(`ship_gate_pass`). Inactive families fall back to the structural model — the
gate is the whole point. Feature vectors must match ml/train_all.py FEATURES.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURES = ["home_for", "home_against", "away_for", "away_against",
            "home_goals", "away_goals", "lam_struct", "threshold",
            "struct_prob", "mkt"]
FAMILIES = ["goals", "corners", "cards", "sot", "fouls"]


class MicroMLModel:
    def __init__(self, path: str):
        self.active = False
        self.model = self.calibrator = None
        self.features = FEATURES
        self.gate: dict = {}
        if Path(path).exists():
            try:
                import joblib
                blob = joblib.load(path)
                if blob.get("ship_gate_pass"):
                    self.model = blob["model"]
                    self.calibrator = blob["calibrator"]
                    self.features = blob.get("features", FEATURES)
                    self.gate = blob.get("walk_forward", {})
                    self.active = True
            except Exception as e:               # noqa: BLE001
                logger.warning("ML load failed (%s): %s", path, e)

    def prob_over(self, feats: dict) -> Optional[float]:
        if not self.active:
            return None
        import pandas as pd
        x = pd.DataFrame([{f: feats.get(f, np.nan) for f in self.features}])
        p = float(self.model.predict_proba(x)[:, 1][0])
        p = float(self.calibrator.predict(np.array([p]))[0])
        return min(max(p, 0.01), 0.99)


class MLRegistry:
    """Loads every family model; .get(family) returns an active model or None."""

    def __init__(self, params_dir: str):
        self.models: Dict[str, MicroMLModel] = {}
        active = []
        for fam in FAMILIES:
            m = MicroMLModel(str(Path(params_dir) / f"ml_{fam}.joblib"))
            self.models[fam] = m
            if m.active:
                active.append(fam)
        if active:
            logger.info("ML models active: %s", active)

    def get(self, family: str) -> Optional[MicroMLModel]:
        m = self.models.get(family)
        return m if (m and m.active) else None
