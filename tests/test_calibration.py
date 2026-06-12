import numpy as np
import pytest

from src.calibration_layer import CalibrationLayer, CalibrationTrainer


def _overconfident_data(n=2000, seed=3):
    """Model says p but truth is squashed toward 0.5 — classic overconfidence."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.05, 0.95, n)
    true = 0.5 + 0.6 * (raw - 0.5)
    outcomes = (rng.random(n) < true).astype(float)
    return raw, outcomes


def test_isotonic_improves_brier():
    raw, outcomes = _overconfident_data()
    cal = CalibrationTrainer().train(raw, outcomes)
    raw2, outcomes2 = _overconfident_data(seed=4)
    fixed = cal.predict(np.asarray(raw2))
    assert np.mean((fixed - outcomes2) ** 2) < np.mean((raw2 - outcomes2) ** 2)


def test_platt_used_below_min_n():
    raw, outcomes = _overconfident_data(n=200)
    cal = CalibrationTrainer().train(raw, outcomes)
    from src.calibration_layer import PlattCalibrator
    assert isinstance(cal, PlattCalibrator)


def test_layer_identity_fallback():
    layer = CalibrationLayer()
    assert layer.calibrate(0.62, "CORNER_MARKET") == 0.62


def test_layer_one_dimensional_input():              # B10 — sklearn shape bug
    raw, outcomes = _overconfident_data()
    cal = CalibrationTrainer().train(raw, outcomes)
    layer = CalibrationLayer({"GOAL_MARKET": cal})
    out = layer.calibrate(0.9, "GOAL_MARKET")
    assert 0.0 < out < 1.0 and out < 0.9             # overconfidence pulled in


def test_ece_zero_for_perfect():
    probs = np.array([0.2] * 500 + [0.8] * 500)
    rng = np.random.default_rng(0)
    outcomes = (rng.random(1000) < probs).astype(float)
    assert CalibrationTrainer.ece(probs, outcomes) < 0.05
