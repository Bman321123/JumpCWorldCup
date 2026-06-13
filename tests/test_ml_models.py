"""ML micro-model gating — a failed-gate model must stay inactive and be a no-op."""
from pathlib import Path

from src.ml_models import CornerMLModel

ROOT = Path(__file__).resolve().parents[1]


def test_failed_gate_model_is_inactive():
    """ml_corners.joblib failed its gate, so the wrapper must be inactive and
    prob_over must return None (orchestrator then keeps the structural model)."""
    m = CornerMLModel(str(ROOT / "params" / "ml_corners.joblib"))
    assert m.active is False
    assert m.prob_over(5, 5, 5, 5, 10.0, 9.5, 0.5) is None


def test_missing_model_is_inactive():
    m = CornerMLModel("/nonexistent/path.joblib")
    assert m.active is False
    assert m.prob_over(5, 5, 5, 5, 10.0, 9.5, 0.5) is None
