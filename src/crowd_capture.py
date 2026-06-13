"""Crowd-forecast capture: parse question blocks scraped from the platform page
(by extension/), store them, and fuzzy-join them to our predictions.

The contest's own questions + crowd numbers are the most relevant dataset that
exists (ROADMAP 1.5) — they drive the submission policy (crowd-relative RBP)
and, accumulated, the crowd-bias model.
"""
from __future__ import annotations

import difflib
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

QUESTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Q\d+\W*)?"
    r"((?:At [A-Za-z\- ]+,? )?(?:Will|Who|Which|What|How|Does|Do|Is|Are|Can)\b"
    r"[^?\n]{10,250}\?)", re.IGNORECASE)
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
CROWD_HINTS = ("crowd", "community", "consensus", "average", "forecasters",
               "users", "public", "field")
OWN_HINTS = ("your", "you:", "you ", "my ", "submitted")

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crowd_capture (
    question_norm TEXT,
    capture_day TEXT,
    question_text TEXT,
    crowd_pct REAL,
    own_pct REAL,
    ambiguous INT,
    url TEXT,
    raw_block TEXT,
    captured_at TEXT,
    PRIMARY KEY (question_norm, capture_day) ON CONFLICT REPLACE
);
"""


def normalize_question(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def parse_block(block: str) -> Optional[dict]:
    """One captured DOM block -> {question_text, crowd_pct, own_pct, ambiguous}.
    Returns None if no question is present. Defensive: the platform's exact
    format is unknown, so prefer storing with ambiguous=1 over dropping data."""
    qm = QUESTION_RE.search(block)
    if not qm:
        return None
    question = re.sub(r"\s+", " ", qm.group(1)).strip()

    crowd = own = None
    unlabeled: List[float] = []
    for line in block.splitlines():
        for m in PCT_RE.finditer(line):
            val = float(m.group(1))
            if not (0 <= val <= 100):
                continue
            low = line.lower()
            if any(h in low for h in CROWD_HINTS) and crowd is None:
                crowd = val
            elif any(h in low for h in OWN_HINTS) and own is None:
                own = val
            else:
                unlabeled.append(val)
    ambiguous = 0
    if crowd is None:
        if len(unlabeled) == 1:
            crowd = unlabeled[0]
        elif unlabeled:
            crowd = unlabeled[0]
            ambiguous = 1
    if crowd is None:
        return None
    return {"question_text": question, "crowd_pct": crowd, "own_pct": own,
            "ambiguous": ambiguous, "raw_block": block}


def parse_blocks(blocks: List[str]) -> List[dict]:
    out, seen = [], set()
    for b in blocks:
        row = parse_block(b)
        if row:
            norm = normalize_question(row["question_text"])
            if norm not in seen:
                seen.add(norm)
                out.append(row)
    return out


def store_capture(db_path: str, rows: List[dict], url: str = "") -> int:
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(db_path)
    con.executescript(TABLE_SQL)
    for r in rows:
        con.execute(
            "INSERT OR REPLACE INTO crowd_capture (question_norm, capture_day, "
            "question_text, crowd_pct, own_pct, ambiguous, url, raw_block, "
            "captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (normalize_question(r["question_text"]), now.date().isoformat(),
             r["question_text"], r["crowd_pct"], r.get("own_pct"),
             r.get("ambiguous", 0), url, r.get("raw_block", ""),
             now.isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return len(rows)


def latest_crowd(db_path: str, hours: float = 36.0) -> Dict[str, dict]:
    """question_norm -> {question_text, crowd_pct, ambiguous} from recent captures."""
    con = sqlite3.connect(db_path)
    con.executescript(TABLE_SQL)
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    out = {}
    for norm, text, pct, amb, ts in con.execute(
            "SELECT question_norm, question_text, crowd_pct, ambiguous, "
            "captured_at FROM crowd_capture"):
        try:
            t = datetime.fromisoformat(ts).timestamp()
        except ValueError:
            continue
        if t >= cutoff:
            out[norm] = {"question_text": text, "crowd_pct": pct,
                         "ambiguous": amb}
    con.close()
    return out


def fuzzy_lookup(question_text: str, crowd: Dict[str, dict],
                 min_ratio: float = 0.93) -> Optional[dict]:
    norm = normalize_question(question_text)
    if norm in crowd:
        return crowd[norm]
    best, best_ratio = None, min_ratio
    for cand_norm, row in crowd.items():
        ratio = difflib.SequenceMatcher(None, norm, cand_norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = row, ratio
    return best
