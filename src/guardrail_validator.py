"""Coherence guardrails over a match's full question set (PRD v2.2 §6.8).

Runs after blending, before clipping:
1. 1X2 trio renormalized to sum to 1
2. Threshold ladders monotone (P(over 1.5) >= P(over 2.5)) via PAV projection
3. GTE/LT complements; P(advance) >= P(win 90) per team
4. Clip to [0.001, 0.999] LAST
Corrections > 0.02 are logged — they signal upstream disagreement.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

from .count_math import clip_prob
from .types import Condition, ParsedQuestion, Prediction, ResultScope

logger = logging.getLogger(__name__)

FLOOR, CEILING = 0.001, 0.999
WARN_DELTA = 0.02


def pav_non_increasing(values: Sequence[float]) -> List[float]:
    """Project a sequence onto the nearest non-increasing sequence (PAV on the
    negated series)."""
    v = [-x for x in values]
    # classic pool-adjacent-violators for non-decreasing on v
    blocks = [[x, 1.0] for x in v]  # [mean, weight]
    out: List[List[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2 = out.pop()
            m1, w1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    result: List[float] = []
    for mean, weight in out:
        result.extend([-mean] * int(round(weight)))
    return result


class GuardrailValidator:
    def normalize_trio(self, probs: Dict[str, float]) -> Dict[str, float]:
        total = sum(probs.values())
        if total <= 0:
            raise ValueError(f"Degenerate trio: {probs}")
        return {k: v / total for k, v in probs.items()}

    def enforce_bounds(self, p: float, context: str = "") -> float:
        clipped = clip_prob(p, FLOOR, CEILING)
        if abs(clipped - p) > 1e-12:
            logger.warning("Clipped %.6f -> %.3f (%s)", p, clipped, context)
        return clipped

    def check_joint_leq_marginal(self, joint_p: float, marginal_p: float,
                                 desc: str = "") -> bool:
        ok = joint_p <= marginal_p + 1e-3
        if not ok:
            logger.critical("Monotonicity violation: joint %.4f > marginal %.4f (%s)",
                            joint_p, marginal_p, desc)
        return ok

    def validate_match_set(self, predictions: List[Prediction],
                           parsed: List[ParsedQuestion]) -> List[Prediction]:
        by_id = {q.question_id: q for q in parsed}

        # --- 1X2 trio renormalization (WIN_90 entries only) ---
        trio_idx = {}
        for i, pred in enumerate(predictions):
            q = by_id.get(pred.question_id)
            if q and q.family.value == "MATCH_RESULT" and q.scope != ResultScope.ADVANCE:
                trio_idx[q.target] = i
        if {"HOME", "DRAW", "AWAY"} <= set(trio_idx):
            raw = {t: predictions[i].p_blend for t, i in trio_idx.items()}
            norm = self.normalize_trio(raw)
            for t, i in trio_idx.items():
                if abs(norm[t] - raw[t]) > WARN_DELTA:
                    logger.warning("Trio renorm moved %s by %.3f", t, norm[t] - raw[t])
                predictions[i].p_blend = norm[t]

        # --- threshold-ladder monotonicity per (family, metric, target, window, GTE) ---
        ladders: Dict[tuple, List[int]] = {}
        for i, pred in enumerate(predictions):
            q = by_id.get(pred.question_id)
            if q and q.condition == Condition.GTE:
                key = (q.family.value, q.metric, q.target, q.window.value)
                ladders.setdefault(key, []).append(i)
        for key, idxs in ladders.items():
            if len(idxs) < 2:
                continue
            idxs.sort(key=lambda i: by_id[predictions[i].question_id].threshold)
            vals = [predictions[i].p_blend for i in idxs]
            fixed = pav_non_increasing(vals)
            for i, v_new, v_old in zip(idxs, fixed, vals):
                if abs(v_new - v_old) > WARN_DELTA:
                    logger.warning("Ladder %s adjusted %.3f -> %.3f", key, v_old, v_new)
                predictions[i].p_blend = v_new

        # --- advance >= win90 for the same side ---
        adv = {}
        win90 = {}
        for i, pred in enumerate(predictions):
            q = by_id.get(pred.question_id)
            if q and q.family.value == "MATCH_RESULT" and q.target in ("HOME", "AWAY"):
                (adv if q.scope == ResultScope.ADVANCE else win90)[q.target] = i
        for side in ("HOME", "AWAY"):
            if side in adv and side in win90:
                if predictions[adv[side]].p_blend < predictions[win90[side]].p_blend:
                    logger.warning("P(advance) < P(win90) for %s; raising to match.", side)
                    predictions[adv[side]].p_blend = predictions[win90[side]].p_blend

        # --- clip LAST ---
        for pred in predictions:
            pred.p_final = self.enforce_bounds(pred.p_blend, pred.question_id)
        return predictions
