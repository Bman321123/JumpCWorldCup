# ROADMAP — from working pipeline to the full vision

The target end product, per match, within 60 seconds of questions posting:

```
For every question:
  1. Devigged sharp-book probability        (Pinnacle anchor — LIVE today)
  2. ML model probability                   (gradient boosting over the full
                                             feature store, calibrated)
  3. Structural model fallback              (Dixon-Coles & friends — LIVE today)
  4. Final submission                       (crowd-relative policy: how far from
                                             the crowd to stand = our bet sizing)
  5. Edge report                            (tonight's questions ranked by
                                             model-vs-crowd disagreement)
```

The contest's version of "placing a wager" is **how far from the crowd we submit**.
We are rewarded for being right where the crowd is wrong — in either direction —
and penalized quadratically for being bold and wrong. Boldness is the *output* of
edge, never the input.

---

## Phase 1 — Data foundation (June 12–16, group stage matchday 1–2)

The ML model is only as good as what it eats. Everything below is public and free.

| # | Task | Source | Status |
|---|------|--------|--------|
| 1.1 | Per-team micro rates: corners for/against, cards, fouls, offsides, SOT — last 4 years, all 48 teams + historical label corpus (WC 2014/18/22, Euro, Copa, qualifiers) | FBref match reports (polite overnight scrape; scaffold exists in `ingestion/ingest_fbref.py`) | ☐ |
| 1.2 | Club-football training corpus: ~40k matches with corners, cards, AND closing odds (market features + labels at volume) | football-data.co.uk CSVs (instant download) | ☐ |
| 1.3 | Referee card table: 2026 referee list × career rates | FIFA.com + worldfootball.net | ☐ |
| 1.4 | Player shares for ~200 likely prop subjects: xG+xA involvement, SOT/90, expected minutes | FIFA squad lists + FBref player pages | ☐ |
| 1.5 | **Crowd capture**: log question text, crowd %, our submission, outcome for every match — the contest's own 1,040 questions are the most relevant training data that exists, and nobody else is logging it | Manual paste after each match (CLI) or repurpose the CoreProp Chrome extension to read the platform page | ☐ |
| 1.6 | Pin DraftKings event-group id + FanDuel page id; verify both scrapers live (redundancy for Pinnacle) | Browser devtools, ~30 min | ☐ |

## Phase 2 — ML model v2 (June 16–22)

**How it works, for a first ML build:** a gradient-boosted tree model is a few
hundred small decision trees voting in sequence, each one correcting the errors of
the ones before. We train one per question family. We feed it ~90 features per
match-question and the binary outcome; it learns the weightings automatically —
that's the "all of that data weighted properly" you described. Our job is not to
hand-tune weights; it is to (a) feed it clean honest features, (b) never let it
see the future during training, (c) verify it on matches it has never seen.

- 2.1 Feature store v2 — the full inventory (PRD §8.3): team form/strength, the
      Phase-1 micro rates (opponent-adjusted), player aggregates (expected-XI
      involvement, star concentration, absences), context (weather at venue +
      kickoff hour, altitude, referee, rest days, motivation state), question
      descriptors (threshold, half, comparative/compound flags).
- 2.2 **Devigged Pinnacle lines as input features.** The model then learns the
      *residual* — where sharp markets tend to be slightly off for corners/cards
      (the markets are thin there) — instead of re-deriving football from scratch.
      This is the single highest-value feature group.
- 2.3 Training discipline (already built, reused): walk-forward splits only,
      monotone constraints, Optuna tuning, isotonic calibration per family,
      internationals-only recalibration on top.
- 2.4 **Ship gate per family** (already built): the ML deploys only where it beats
      the structural+market blend out-of-sample. Batch-1 failed this gate and was
      correctly benched; Phase-1 data + market features are what batch 2 needs to
      pass it. Families that never pass keep running the structural logic — that
      is a feature, not a failure.

## Phase 3 — Decision layer (June 20–26)

- 3.1 **Edge report** per match: every question with market prob, ML prob,
      structural prob, crowd (when visible), and the recommended submission —
      ranked by expected RBP. This is the "spit out both odds" deliverable; the
      manifests already carry model vs market side by side, this formats it for
      decision-making.
- 3.2 Auto-λ: per-family model-error (τ) re-estimated weekly from
      `predictions_log` outcomes instead of priors.
- 3.3 Leaderboard position → κ (variance appetite) fed into the submission policy.
- 3.4 Hard rails stay non-negotiable: 0.85 player-prop cap, [0.03, 0.97] bounds,
      ladder/trio coherence. The -42s are the thing that loses this contest.

## Phase 4 — Tournament ops (continuous; FREEZE June 27)

- Nightly: refit Dixon-Coles + micro rates with new matches (cron exists).
- Weekly + once before R32: ML retrain including in-tournament questions — by the
  knockouts the model has seen ~700 real platform questions no competitor model has.
- After every matchday: calibration report + crowd-bias report (where is the crowd
  systematically wrong? early evidence: compound questions and player props).
- **June 27 freeze**: knockouts run validated config only. R32 weight is 2×,
  final is ~16× — no live experiments after group stage.

## Division of labor

**Code, scrapes, training, backtests, reports:** the bot (me).
**You:** confirm platform rules (§0.4 — especially whether knockout "win" means
advance), paste crowd numbers until the extension exists, check lineups at T-75min,
and physically submit. Submission stays human — at 16× weight the human is the
final guardrail.
