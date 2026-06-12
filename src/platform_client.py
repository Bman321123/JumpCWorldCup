"""SportsPredict platform client — the official bot API (MCP over HTTP).

Capabilities (confirmed live 2026-06-13): list events/lobbies/matches/markets,
submit predictions (integer 1-99), UPDATE predictions before kickoff, and pull
settled results with per-prediction Brier scores. Crowd probabilities are NOT
exposed here — the browser extension (extension/) remains the crowd source.

Key comes from SPORTSPREDICT_API_KEY (env or .env). Never hardcode it.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

MCP_URL = "https://api.sportspredict.com/api/v1/mcp"
PROTOCOL = "2025-03-26"


def _load_key() -> str:
    key = os.environ.get("SPORTSPREDICT_API_KEY")
    if not key:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SPORTSPREDICT_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("SPORTSPREDICT_API_KEY not set (env or .env)")
    return key


class PlatformClient:
    def __init__(self, api_key: Optional[str] = None):
        self.key = api_key or _load_key()
        self.session_id: Optional[str] = None
        self._rid = 0

    # ----- MCP plumbing -----

    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.key}",
             "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def _rpc(self, method: str, params: dict) -> dict:
        self._rid += 1
        r = requests.post(MCP_URL, headers=self._headers(), timeout=30,
                          json={"jsonrpc": "2.0", "id": self._rid,
                                "method": method, "params": params})
        r.raise_for_status()
        sid = r.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        data = None
        for line in r.text.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
        if data is None and r.text.strip():
            data = json.loads(r.text)
        if data and "error" in data:
            raise RuntimeError(f"MCP error on {method}: {data['error']}")
        return (data or {}).get("result", {})

    def connect(self) -> None:
        self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                 "clientInfo": {"name": "jumpcworldcup",
                                                "version": "1.0"}})

    def call(self, tool: str, args: dict) -> dict:
        if self.session_id is None:
            self.connect()
        result = self._rpc("tools/call", {"name": tool, "arguments": args})
        # tool results arrive as content blocks; JSON payloads as text
        out = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    out.append(json.loads(block["text"]))
                except (json.JSONDecodeError, TypeError):
                    out.append(block["text"])
        if len(out) == 1:
            return out[0]
        return {"blocks": out, "raw": result}

    # ----- platform operations -----

    def list_events(self, limit: int = 20):
        return self.call("list_events", {"limit": limit})

    def list_lobbies(self, event_id):
        return self.call("list_lobbies", {"event_id": event_id})

    def join_lobby(self, lobby_id, password: Optional[str] = None):
        args = {"lobby_id": lobby_id}
        if password:
            args["password"] = password
        return self.call("join_lobby", args)

    def list_matches(self, event_id=None, lobby_id=None):
        args = {}
        if event_id is not None:
            args["event_id"] = event_id
        if lobby_id is not None:
            args["lobby_id"] = lobby_id
        return self.call("list_matches", args)

    def list_markets(self, lobby_id, match_id=None):
        args = {"lobby_id": lobby_id}
        if match_id is not None:
            args["match_id"] = match_id
        return self.call("list_markets", args)

    def submit_batch(self, predictions: List[dict]):
        """predictions: [{market_id, lobby_id, probability(int 1-99)}].
        OUTWARD-FACING: only call via tools/submit.py with explicit human go."""
        return self.call("submit_predictions_batch", {"predictions": predictions})

    def update_prediction(self, prediction_id, probability: int):
        return self.call("update_prediction", {"prediction_id": prediction_id,
                                               "probability": probability})

    def list_predictions(self, lobby_id=None):
        args = {"lobby_id": lobby_id} if lobby_id else {}
        return self.call("list_predictions", args)

    def list_results(self, lobby_id=None):
        args = {"lobby_id": lobby_id} if lobby_id else {}
        return self.call("list_results", args)


def to_platform_probability(p: float) -> int:
    """Pipeline [0,1] float -> platform integer percent, clamped to 1-99."""
    return int(min(max(round(p * 100), 1), 99))
