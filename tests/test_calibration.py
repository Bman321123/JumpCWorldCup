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
    from src.calibration_layer import PlattCalibrator, RangeGuardedCalibrator
    assert isinstance(cal, RangeGuardedCalibrator)
    assert isinstance(cal.inner, PlattCalibrator)


def test_layer_identity_fallback():
    layer = CalibrationLayer()
    assert layer.calibrate(0.62, "CORNER_MARKET") == 0.62


def test_layer_one_dimensional_input():              # B10 — sklearn shape bug
    raw, outcomes = _overconfident_data()
    cal = CalibrationTrainer().train(raw, outcomes)
    layer = CalibrationLayer({"GOAL_MARKET": cal})
    out = layer.calibrate(0.9, "GOAL_MARKET")
    assert 0.0 < out < 1.0 and out < 0.9             # overconfidence pulled in


def test_range_guard_identity_outside_training_support():
    """Live bug 2026-06-13: a Platt calibrator trained on probs in ~[0.35,0.65]
    mapped a sane 0.23 compound estimate to 0.42. Outside its training range a
    calibrator must be identity."""
    rng = np.random.default_rng(5)
    raw = rng.uniform(0.35, 0.65, 400)            # narrow training band
    outcomes = (rng.random(400) < raw).astype(float)
    cal = CalibrationTrainer().train(raw, outcomes)
    layer = CalibrationLayer({"GOAL_MARKET": cal})
    assert layer.calibrate(0.10, "GOAL_MARKET") == pytest.approx(0.10)
    assert layer.calibrate(0.23, "GOAL_MARKET") == pytest.approx(0.23)
    assert layer.calibrate(0.92, "GOAL_MARKET") == pytest.approx(0.92)
    # inside the band it still calibrates (output may differ from input)
    inside = layer.calibrate(0.50, "GOAL_MARKET")
    assert 0.3 < inside < 0.7


def test_ece_zero_for_perfect():
    probs = np.array([0.2] * 500 + [0.8] * 500)
    rng = np.random.default_rng(0)
    outcomes = (rng.random(1000) < probs).astype(float)
    assert CalibrationTrainer.ece(probs, outcomes) < 0.05
