"""ML micro-model gating — failed-gate families must stay inactive (no-op)."""
from pathlib import Path

from src.ml_models import FAMILIES, MicroMLModel, MLRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_missing_model_is_inactive():
    m = MicroMLModel("/nonexistent/path.joblib")
    assert m.active is False
    assert m.prob_over({"home_for": 5, "threshold": 9.5}) is None


def test_registry_loads_all_families():
    reg = MLRegistry(str(ROOT / "params"))
    # every family slot exists; .get returns a model only if it passed its gate
    for fam in FAMILIES:
        assert fam in reg.models
        got = reg.get(fam)
        assert got is None or got.active is True


def test_failed_gate_family_returns_none():
    """A trained-but-failed family (e.g. corners) must not be served."""
    reg = MLRegistry(str(ROOT / "params"))
    corners = reg.models.get("corners")
    if corners is not None and not corners.active:
        assert reg.get("corners") is None
