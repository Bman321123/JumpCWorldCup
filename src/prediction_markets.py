"""Prediction-market cross-check (Polymarket public Gamma API).

Sanity input only (PRD §7) — printed beside our numbers for big matches, never
wired into the blend automatically. Degrades to empty list on any failure.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"


def search_polymarket(query: str, limit: int = 10) -> List[dict]:
    """Best-effort search. Returns [{'question', 'yes_price'}]."""
    try:
        import requests
        r = requests.get(GAMMA_URL,
                         params={"closed": "false", "limit": limit, "search": query},
                         timeout=8)
        r.raise_for_status()
        out = []
        for m in r.json():
            prices = m.get("outcomePrices")
            question = m.get("question")
            if not question or not prices:
                continue
            try:
                if isinstance(prices, str):
                    import json as _json
                    prices = _json.loads(prices)
                yes = float(prices[0])
            except (ValueError, IndexError, TypeError):
                continue
            out.append({"question": question, "yes_price": yes})
        return out
    except Exception as e:                       # noqa: BLE001
        logger.warning("Polymarket lookup failed: %s", e)
        return []


def crosscheck_lines(home_name: str, away_name: str) -> List[str]:
    notes = []
    for hit in search_polymarket(f"{home_name} {away_name}"):
        notes.append(f"[polymarket] {hit['question']}: yes={hit['yes_price']:.3f}")
    return notes
