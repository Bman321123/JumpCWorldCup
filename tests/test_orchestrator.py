"""End-to-end smoke test: offline pipeline on a synthetic parameter set."""
import json
from pathlib import Path

import pytest

from src.orchestrator import Orchestrator
from src.stats_engine import ModelParameters

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    "Will Argentina win?",
    "Will the match end in a draw?",
    "Will France win?",
    "Will there be over 2.5 goals?",
    "Will both teams score?",
    "Will there be 10 or more corners?",
    "Will there be 4 or more yellow cards?",
    "Will Argentina be caught offside 2 or more times in the first half?",
    "What is the meaning of life?",            # must hit the fallback path
]


@pytest.fixture
def orch(tmp_path):
    params = ModelParameters(
        mu=0.18, gamma=0.25, rho=-0.12,
        attack={"ARG": 0.45, "FRA": 0.42}, defense={"ARG": 0.40, "FRA": 0.38})
    p = tmp_path / "params.json"
    params.save(str(p))
    return Orchestrator(config_dir=str(ROOT / "config"), params_path=str(p),
                        online=False)


def test_full_match_offline(orch, tmp_path):
    # group stage: "win" = WIN_90, so the H/D/A trio must sum to 1
    manifest = orch.predict_match("ARG", "FRA", "2026-06-20", QUESTIONS,
                                  tournament_round="group",
                                  output_dir=str(tmp_path / "out"))
    preds = manifest["predictions"]
    assert len(preds) == len(QUESTIONS)
    for p in preds:
        assert 0.001 <= p["final_probability"] <= 0.999
    # trio coherence after guardrails
    trio = [p["final_probability"] for p in preds[:3]]
    assert sum(trio) == pytest.approx(1.0, abs=0.01)
    # fallback question flagged for review, not silently guessed
    fb = preds[-1]
    assert fb["source"] == "fallback"
    assert "review" in fb["notes"].lower()
    # manifest written
    files = list((tmp_path / "out").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["match_id"] == "ARG_v_FRA_2026-06-20"


def test_knockout_trio_not_renormalized(orch):
    """In a knockout, ADVANCE(home) + draw90 + ADVANCE(away) legitimately
    exceeds 1 — the trio renorm must only bind WIN_90 entries."""
    manifest = orch.predict_match("ARG", "FRA", "2026-07-19", QUESTIONS[:3],
                                  tournament_round="final")
    assert manifest["round_weight"] == 16.0
    total = sum(p["final_probability"] for p in manifest["predictions"])
    assert total > 1.05


def test_knockout_win_uses_advance(orch):
    manifest = orch.predict_match("ARG", "FRA", "2026-07-19",
                                  ["Will Argentina win?",
                                   "Will Argentina win in 90 minutes?"],
                                  tournament_round="final")
    p_adv = manifest["predictions"][0]["final_probability"]
    p_90 = manifest["predictions"][1]["final_probability"]
    assert p_adv >= p_90                              # B8


def test_market_probs_never_calibrated(tmp_path):     # B5 — pipeline wiring
    """The market branch must bypass the calibrator entirely."""
    import numpy as np
    from src.calibration_layer import CalibrationLayer, CalibrationTrainer

    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.95, 1000)
    outcomes = (rng.random(1000) < 0.5 + 0.4 * (raw - 0.5)).astype(float)
    cal = CalibrationTrainer().train(raw, outcomes)

    params = ModelParameters(attack={"ARG": 0.4, "FRA": 0.4},
                             defense={"ARG": 0.4, "FRA": 0.4})
    pfile = tmp_path / "p.json"
    params.save(str(pfile))
    orch = Orchestrator(config_dir=str(Path(__file__).resolve().parents[1] / "config"),
                        params_path=str(pfile), online=False)
    orch.calibration = CalibrationLayer({"GOAL_MARKET": cal})

    from src.question_classifier import QuestionClassifier
    q = orch.classifier.parse("Will there be over 2.5 goals?", "ARG", "FRA", "group")
    market = {"is_closing": True, "h2h": None, "totals": {2.5: 0.61}, "btts": None}
    from src.types import MatchContext
    pred = orch._predict_one(q, MatchContext("ARG", "FRA", "2026-06-20"), market)
    assert pred.p_market == 0.61                      # untouched by the calibrator
    assert pred.p_model_cal != pred.p_model_raw       # model branch WAS calibrated
