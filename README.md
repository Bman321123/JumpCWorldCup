# JumpCWorldCup — Probability Cup Forecasting System

Forecasting system for the **Jump Trading Probability Cup** (sportspredict.com/probabilitycup):
calibrated probabilities for ~10 binary questions per match across all 104 matches of
the 2026 FIFA World Cup, scored by weighted Brier score. Built from
`PRD_JumpProbabilityCup_v2.md` (v2.2) — see that document for the full design rationale
and the changelog of v1.0 bugs this implementation fixes.

## Architecture

```
parse question ─► resolve context (referee / altitude / weather / motivation MC / absences)
   ├─ market branch:  The Odds API ─► sharp-book selection ─► Shin devig ────────────┐
   └─ model branch:   Dixon-Coles score matrix / NB cards / Poisson micro-markets    │
                      ─► per-family calibrator (model branch ONLY) ──────────────────┤
                                                                                     ▼
                       logit-space blend ─► coherence guardrails ─► clip ─► manifest + log
```

Key properties (each is a pinned regression test in `tests/`):

- **Threshold semantics**: `ceil()`-based — "over 2.5" = P(≥3), GTE/LT are exact complements
- **One τ-corrected score matrix** drives 1X2, totals, BTTS, team totals, clean sheets
- **ET/penalties layer**: knockout "win" questions default to ADVANCE scope
- **Markets are never calibrated** — calibration applies to the model branch only
- **Coherence guardrails**: threshold ladders monotone (PAV), 1X2 trio sums to 1,
  P(advance) ≥ P(win 90'), clip to [0.001, 0.999] last
- **Motivation Monte Carlo**: must-win/safe/eliminated states (dead rubbers *lower*
  card intensity)
- **Player layer**: availability multipliers from involvement shares, suspension
  tracking from accumulated yellows, beta-binomial shrinkage everywhere

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Historical data (martj42 international results, 1872–present)
python ingestion/ingest_historical.py

# 2. Optional: Elo priors for sparse teams
python ingestion/ingest_elo.py

# 3. Fit Dixon-Coles (time-decayed, ridge toward Elo priors)
python ingestion/compute_parameters.py --since 2019-01-01 --elo params/elo.json

# 4. Verify
pytest -q

# 5. Predict a match (offline = pure model; drop --offline once ODDS_API_KEY is set)
python cli/predict.py --home MEX --away RSA --date 2026-06-11 --round group \
  --stadium "Estadio Azteca" --offline \
  --questions "Will Mexico win?; Will there be over 2.5 goals?; Will both teams score?; Will there be 10 or more corners?; Will there be 4 or more yellow cards?"
```

Outputs land in `output/predictions/{match_id}.json` plus a human-readable
`_summary.txt`. **Review every number before submitting** — at knockout weights the
human is the last guardrail.

## Market data

Set `ODDS_API_KEY` (the-odds-api.com; paid tier recommended — one request costs
`markets × regions` credits). The client prefers sharp books (Pinnacle, exchanges),
falls back to a median-of-books consensus, Shin-devigs everything, and caches in
SQLite. `src/prediction_markets.py` adds a Polymarket cross-check (sanity print only,
never auto-blended). No key → the pipeline runs pure-model, which is the designed
fallback for corners/cards/offsides anyway.

## Backtest & ML layer

```bash
python backtest/replay.py --tournaments WC2018,WC2022,EURO2024   # as-of fits, Brier vs base rates
python ml/feature_store.py                                        # as-of question-level rows
python ml/train_gbm.py                                            # walk-forward GBM + ship gate
```

The ship gate is the contract: a family's ML stack deploys only if it beats the
structural baseline on walk-forward weighted Brier. Otherwise that family stays on
structural logic.

## Day-0 checklist (verify on the platform — PRD §0.4)

- [ ] Actual round-weight multipliers → edit `config/round_weights.json` (currently ASSUMED 1/2/4/8/12/16)
- [ ] How unanswered questions are scored (skip policy changes strategy)
- [ ] Whether knockout "Will X win?" means 90 minutes or advance → flip default in `src/question_classifier.py`
- [ ] Whether entries can be revised before kickoff (enables the T-30min closing-line re-run)
- [ ] Whether player-level questions are asked (activates `player_layer.player_prop`)

## What still needs data (Tier 1/2 work)

- `config/referee_table.json` — populate from FIFA's 2026 referee list (multipliers
  shrink to 1.0 when empty, so this is safe to defer)
- `config/player_shares.json` — involvement shares via `ingestion/ingest_fbref.py`
- Team corner/card/offside rates — FBref tournament match reports (engine uses sane
  global defaults until then)
- Re-fit nightly during the tournament: `python ingestion/compute_parameters.py`

## Repository layout

```
src/          pipeline modules (types, devig, engine, classifier, calibration,
              blender, guardrails, context, players, odds, orchestrator, db)
ingestion/    data loaders: martj42 results, Elo, FBref scaffold, parameter fit
ml/           as-of feature store + walk-forward GBM with ship gates
backtest/     as-of tournament replays vs base-rate baselines
cli/          predict.py — per-match entry point
config/       groups (real draw), venues (16), round weights, referees, player shares
params/       fitted artifacts (dixon_coles.json, elo.json, ml models)
tests/        pinned regression tests for every v1.0 bug (B1–B10)
```
