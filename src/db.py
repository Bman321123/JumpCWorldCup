"""SQLite schema (PRD v2.2 §7)."""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    date TEXT, home_team TEXT, away_team TEXT,
    home_score INT, away_score INT,
    tournament TEXT, city TEXT, country TEXT, neutral INT,
    home_corners INT, away_corners INT,
    home_yellows INT, away_yellows INT,
    home_reds INT, away_reds INT,
    home_offsides INT, away_offsides INT
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);

CREATE TABLE IF NOT EXISTS predictions_log (
    prediction_id TEXT PRIMARY KEY,
    match_id TEXT, question_id TEXT, question_text TEXT, question_family TEXT,
    p_model_raw REAL, p_model_cal REAL, p_market REAL, p_blend REAL,
    submitted_probability REAL, closing_market_probability REAL,
    round_weight REAL, actual_outcome INT, brier_contribution REAL,
    source TEXT, submitted_at TEXT
);

CREATE TABLE IF NOT EXISTS referee_stats (
    referee_id TEXT PRIMARY KEY,
    name TEXT, nationality TEXT,
    yellow_per_match REAL, red_per_match REAL, fouls_per_match REAL,
    total_matches INT
);

CREATE TABLE IF NOT EXISTS team_rates (
    team_code TEXT PRIMARY KEY,
    attack_strength REAL, defense_strength REAL,
    corner_rate_for REAL, corner_rate_against REAL,
    yellow_rate REAL, red_rate REAL, offside_rate REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS player_cards (
    player TEXT, team TEXT, match_id TEXT,
    yellows INT DEFAULT 0, reds INT DEFAULT 0, wiped INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS market_cache (
    cache_key TEXT PRIMARY KEY,
    raw_json TEXT, fetched_at REAL
);

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


def init_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # migration for DBs created before crowd capture existed
    try:
        con.execute("ALTER TABLE predictions_log ADD COLUMN crowd_probability REAL")
    except sqlite3.OperationalError:
        pass                                     # column already present
    con.commit()
    con.close()
