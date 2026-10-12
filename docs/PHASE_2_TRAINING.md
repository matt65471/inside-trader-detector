# Phase 2 training handoff

This document is the handoff from the completed Phase 1 data-pipeline
implementation to Phase 2 supervised modeling. Phase 2 remains read-only. It must
not place trades, link wallet identities, train a reinforcement-learning agent, or
claim to identify legal insiders.

## Authoritative dataset

- Windows checkout: `C:\Users\matt6\projects\inside-trader-detector`
- SQLite database: `C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.sqlite3`
- Cohort: `balanced-tags-v2`
- Required resolution source: `gamma_markets_v1`
- Scheduled collection task: `InformedFlow-gamma-v2`
- Collection log: `C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.log`
- Python: `C:\Users\matt6\projects\inside-trader-detector\.venv311\Scripts\python.exe`

Do not train from `balanced-tags-v1`, `label_fix.sqlite3`, or the earlier
`test_90d_v2.sqlite3`. Those runs predate the authoritative Gamma resolution gate
or were diagnostic datasets. Do not copy a live SQLite file with an ordinary file
copy while it is being written; stop the task or use SQLite's backup API.

## Phase 1 data lineage

The cohort discovers separate Gamma closed-market feeds for sports, crypto,
weather, culture, finance, geopolitics, and politics, including configured tag
aliases. One page from each active feed is fetched concurrently in stable
round-robin rounds. Each feed has its own durable cursor. Overlapping tag results
deduplicate on `condition_id`, so failure or exhaustion of one feed does not stop
the others.

For each candidate, the pipeline applies deterministic seeded admission. Before
embedding or selection, it performs an exact request to:

```text
GET https://gamma-api.polymarket.com/markets?condition_ids={conditionId}&closed=true&limit=1
```

A market is eligible only when exactly one response row matches the condition ID,
`closed=true`, aligned outcomes/prices/CLOB token arrays are nonempty and unique,
the payout is exactly one-hot (one `1`, all other values `0`), and an explicitly
present `umaResolutionStatus` is `resolved`. The stored resolution contains the
winner index, normalized payouts, aligned token IDs/outcomes, raw Gamma evidence,
and `closedTime` when supplied. Data API `/v2/resolutions` rows cannot pass this
gate.

After Gamma verification, same-event markets and markets with embedding similarity
at least 0.90 to a prior selection are rejected. Selected markets are scanned via
the public Polymarket Data API for all API-served qualifying taker BUY trades worth
at least $100 through the cohort's frozen cutoff. Five-minute crypto Up/Down markets
are excluded. Trade timestamps are enforced locally because some upstream endpoints
do not honor requested bounds.

Enrichment uses public Data API activity from the 30 days strictly before each
trade. Every trade worth at least $500 plus a deterministic 25% sample below $500
is selected for enrichment. The database retains detailed summary JSON and derived
features, not individual raw wallet-history rows. A 100-page cap produces an
explicitly truncated summary rather than pretending the history is complete.

Each qualifying cohort trade is labeled at +900, +3,600, and +86,400 seconds using
the first CLOB price-history point at or after the target and within five minutes.
A public trade print can be used as an approximate fallback. A pre-target point is
never used. Gamma's verified payout supplies `outcome_won` and gross per-share P&L.
Historical fees and net P&L remain unknown unless a time-applicable fee source is
verified. Historical order books and fill depth are unavailable and must not be
inferred from a price-history point.

The durable SQLite queue uses unique job keys, leases, crash recovery, eight
attempts, exponential backoff, and dead-letter reporting. A Gamma HTTP 500 affects
only its feed or condition; downstream 500s become audited incomplete/failed work
without shutting down the full cohort. Scan, enrichment, and labeling handlers
recheck that their trade/market remains selected and Gamma-verified before network
work.

## Current collection snapshot

On 2026-10-08, before the final `finish-existing` report, the database was about
315.06 MB and contained the following cohort counters:

| Category | Eligible | Selected | With trades | Trades | Zero-trade | Redundant | Random rejected | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 4,136 | 239 | 95 | 1,924 | 144 | 1,809 | 2,006 | 761 |
| crypto | 11 | 4 | 2 | 16 | 2 | 4 | 3 | 996 |
| weather | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| culture | 912 | 73 | 50 | 225 | 23 | 365 | 460 | 927 |
| finance | 1,488 | 175 | 150 | 5,255 | 25 | 529 | 782 | 825 |
| geopolitics | 1,100 | 169 | 159 | 5,918 | 10 | 388 | 543 | 831 |
| politics | 655 | 104 | 94 | 7,069 | 10 | 223 | 328 | 896 |

The agreed first modeling scope is finance, geopolitics, politics, and sports.
Together they currently provide:

- 687 selected markets;
- 498 selected markets with at least one qualifying trade; and
- 20,166 qualifying trade rows.

These are provisional counts until `finish-existing` exits and the final report is
captured. Markets/events—not trades—are the closer measure of independent sample
size. Sports currently has only 95 markets with trades, and politics only 94, so
category-specific results will have substantial uncertainty. Crypto, weather, and
culture should remain stored but can be excluded by a training-time category filter.

## Finalization and export

Permanently freeze discovery and finish the selected work from the Windows checkout:

```powershell
Stop-ScheduledTask -TaskName "InformedFlow-gamma-v2"

.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  finish-existing `
  --cohort balanced-tags-v2 `
  --market-workers 2 `
  --enrichment-workers 1 `
  --resolution-workers 1 `
  --label-workers 2
```

The direct command remains attached to the SSH session. For unattended completion,
put that `finish-existing` invocation in `run_gamma_v2.cmd` in place of the `run`
invocation, preserve log redirection, and start the existing scheduled task:

```powershell
Start-ScheduledTask -TaskName "InformedFlow-gamma-v2"
Get-ScheduledTask -TaskName "InformedFlow-gamma-v2" |
  Select-Object TaskName,State
```

Then run and save the report:

```powershell
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  report --cohort balanced-tags-v2 |
  Tee-Object "data\gamma_verified_v2_final_report.txt"
```

Export portable tables:

```powershell
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  export --cohort balanced-tags-v2 --output "data\gamma_verified_v2_export"
```

After the writer has stopped, make a verified SQLite backup before modeling:

```powershell
.\.venv311\Scripts\python.exe -c "import sqlite3; src=sqlite3.connect(r'data\gamma_verified_v2.sqlite3'); dst=sqlite3.connect(r'data\gamma_verified_v2_training_backup.sqlite3'); src.backup(dst); dst.close(); src.close(); print('Backup complete')"
.\.venv311\Scripts\python.exe -c "import sqlite3; db=sqlite3.connect(r'data\gamma_verified_v2_training_backup.sqlite3'); print(db.execute('PRAGMA quick_check').fetchone()[0]); db.close()"
```

The cohort-aware files needed for modeling are:

- `cohort_trades.csv`: trade rows plus canonical category and selection rank.
- `cohort_markets.csv`: the frozen selected-market membership and audit fields.
- `cohort_resolutions.csv`: Gamma winner/payout evidence.
- `cohort_labels.csv`: all three delayed horizons and gross per-share results.
- `model_features.csv`: latest enriched feature row per trade; this file contains
  all database trades, so inner-join it to `cohort_trades.csv` on `trade_key`.
- `wallet_summaries.csv`: detailed 30-day window summaries and coverage metadata.
- `scores.csv`: the existing hand-written heuristic, for comparison only.
- `cohort_similarity_rejections.csv` and `cohort_admission_rejections.csv`: sampling
  audits, not training rows.

The SQLite database remains authoritative. The main trainable view is
`v_model_features`, joined by `trade_key` to cohort membership and one selected
label horizon. It contains trade price, size/notional, market/category fields,
as-of-trade wallet summaries, size relative to wallet median, time to resolution,
recent same-outcome activity, category prior, heuristic score, and historical book
status. Historical book columns will ordinarily be unavailable.

## Required audit before fitting

Do not fit a model until the final report confirms all of the following:

1. Cohort phase is `complete` and discovery is frozen.
2. No relevant jobs remain `pending` or `leased`.
3. Dead/failed/audited-skip jobs have been counted and explained.
4. Every selected market has terminal source `gamma_markets_v1`.
5. Label status and coverage are reported separately for 900, 3,600, and 86,400
   seconds; unavailable labels are never silently treated as losses or zeros.
6. Wallet enrichment coverage is reported as complete, truncated, partial, or
   unavailable; missingness is retained as a feature/quality indicator.
7. Winner class balance, price distribution, market/event concentration, wallet
   concentration, and trades per market are inspected by category and time.
8. Duplicate `trade_key`, condition/outcome mismatches, and label joins are zero.

Record the final counts in this document once the audit is complete. The snapshot
above must not be presented as the final training dataset.

The repository provides a read-only audit command for the authoritative SQLite
backup. It does not initialize, migrate, or otherwise write to the source database:

```powershell
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2_training_backup.sqlite3" `
  audit --cohort balanced-tags-v2 --output "data\gamma_verified_v2_audit"
```

The command writes `balanced-tags-v2_audit.json` for reproducible downstream
checks and `balanced-tags-v2_audit.md` for review. It exits zero only when the
hard integrity gates pass. Legitimately unavailable delayed prices and truncated
wallet histories are reported rather than silently converted into labels or
complete histories.

## Modeling contract

Start with a transparent baseline such as logistic regression, followed only then
by a gradient-boosted tree model if justified. The primary prediction should be a
calibrated probability that the purchased outcome wins, using only information
available at the public trade timestamp. Evaluate the decision to follow at the
chosen delayed entry price—initially +15 minutes—with gross per-share return:

```text
winning trade: 1 - delayed_entry_price
losing trade:  -delayed_entry_price
```

Do not call this net profit because historical fees and executable depth are not
verified. Repeat evaluation at +1 hour and +24 hours as sensitivity analyses.

Use sports, finance, geopolitics, and politics for the first experiment. Preserve
`canonical_category` as a feature or stratification field and always report metrics
separately by category. Because politics and geopolitics contain many trades per
market, either weight markets/events so prolific markets do not dominate or report
both trade-weighted and market-weighted results.

### Split and leakage rules

- Sort by historical time: train on older markets and evaluate on newer markets.
- Keep every trade from the same `condition_id` in one split.
- Keep markets from the same nonempty `event_id`/`event_slug` in one split.
- Fit preprocessing, imputation, and calibration on training data only.
- Features must be values computed as of `trade_ts`. Never use current/lifetime
  wallet performance or later trades as historical features.
- Do not use `outcome_won`, payout vectors, resolution timestamps, label prices, or
  label status as inputs.
- Do not use raw `proxy_wallet`, `condition_id`, `asset_id`, title, slug, or event
  slug as categorical memorization features. Retain them only for grouping/audits.
- Do not use the existing `score`, `bucket`, or `components_json` as ordinary model
  inputs. The heuristic is a benchmark derived from overlapping features, not a
  ground-truth label.
- Select exactly one label horizon per primary experiment. Multiple horizon rows for
  one trade must not be mistaken for independent observations.
- Keep missing/truncated coverage explicit; do not impute it as genuine inactivity.

### Baselines and evaluation

At minimum, compare against:

1. The delayed market price itself as an implied-probability baseline.
2. The observed trade price.
3. The existing heuristic score/bucket.
4. A category-only or constant-probability baseline.

Report sample counts at the trade, wallet, market, and event levels; ROC AUC and PR
AUC; log loss; Brier score; calibration curves/tables; gross P&L per share; and
trade-weighted plus market-weighted results. Break results out by category, price
band, time period, score bucket, wallet-history completeness, and market. Use
bootstrap confidence intervals clustered by event or market. Investigate whether a
few wallets, markets, or longshot wins create the apparent edge.

The model must outperform delayed market price out of sample before discussing an
edge. Paper trading and any action or sizing rule remain separate later phases and
require explicit authorization.
