#!/usr/bin/env bash
# Nightly refresh chain (ROADMAP D). Run overnight during the tournament:
#   bash tools/nightly.sh
# Each step is independent; a failure in one is logged and the chain continues.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG="data/nightly_$(date +%Y%m%d).log"
echo "=== nightly refresh $(date -u +%FT%TZ) ===" | tee -a "$LOG"

run() { echo "--- $* ---" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 || echo "  (failed, continuing)" | tee -a "$LOG"; }

# 1. fresh WC2026 match stats (team + players + referee)
run $PY ingestion/ingest_espn.py --comps WC2026
# 2. recompute per-team micro rates from all match stats
run $PY ingestion/run_fbref_scrape.py --aggregate-only
# 3. refresh player shares + referee table with new matches
run $PY ingestion/build_player_shares.py
run $PY ingestion/build_referee_table.py
# 4. refresh international results, refit Dixon-Coles, refit calibrators
run $PY ingestion/ingest_historical.py
run $PY ingestion/ingest_elo.py
run $PY ingestion/compute_parameters.py --since 2019-01-01 --elo params/elo.json
run $PY calibration/train_calibrators.py
# 5. pull settled platform results into predictions_log (closes the loop)
run $PY tools/sync_results.py

echo "=== done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
