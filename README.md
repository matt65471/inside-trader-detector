# Informed Flow

Informed Flow is a read-only research collector for public Polymarket activity.
It records large public taker buys, builds timestamp-safe market and wallet
features, saves live order-book snapshots, and assigns a transparent heuristic
score. It does **not** connect a wallet, hold credentials, submit orders, or train
a model.

Five-minute crypto “Up or Down” markets are classified before insertion and
hard-excluded from the research tables.

## Requirements

- Python 3.10 or newer
- No third-party runtime dependencies for the core collector; sampled semantic
  selection uses an optional local-model dependency

Run directly from the checkout:

```shell
export PYTHONPATH=src
python -m informed_flow init
python -m informed_flow backfill --days 90
python -m informed_flow sampled-backfill --cohort balanced-lifetime
python -m informed_flow run --cohort balanced-resolved-lifetime
python -m informed_flow watch
```

Alternatively, install the local command in a virtual environment:

```shell
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/informed-flow init
```

On Windows, activate the virtual environment and use
`.venv\Scripts\informed-flow.exe`.

## Commands

All commands accept `--db PATH` before the subcommand. The default database is
`data/informed_flow.sqlite3`.

```shell
python -m informed_flow init
python -m informed_flow backfill --days 90
python -m informed_flow sampled-backfill --cohort balanced-lifetime
python -m informed_flow run --cohort balanced-resolved-lifetime
python -m informed_flow watch --once
python -m informed_flow watch --interval 60
python -m informed_flow enrich
python -m informed_flow rescore
python -m informed_flow report
python -m informed_flow export --output exports
python -m informed_flow retry-jobs
```

For the category-balanced historical cohort, install the optional local semantic
model support first:

```shell
python -m pip install -e ".[semantic]"
python -m informed_flow sampled-backfill --cohort balanced-lifetime \
  --markets-per-category 1000 --admission-rate 0.50 --seed 0 \
  --similarity-threshold 0.90 --market-workers 2 \
  --embedding-device auto
```

`sampled-backfill` resolves Gamma's numeric tag IDs and scans separate closed-market
keyset feeds for sports, crypto, weather, pop-culture, finance, geopolitics, and
politics. Alias feeds merge elections into politics, economy/business into finance,
and world into geopolitics. Each tag has its own durable cursor, so sparse categories
do not wait behind unrelated sports pages and interrupted runs resume each feed
independently. Overlapping tag results are deduplicated locally by condition ID.

Feeds are processed in recent-first API order and stop as soon as every category
reaches its target. A seeded 50% admission decision spreads the sample farther
through recent history; admitted markets are embedded immediately. Markets from
the same event and titles with embedding cosine similarity of at least 0.90 are
rejected, and scanning continues until the quota is filled. If all relevant tag
feeds are exhausted, random rejects are reconsidered in discovery order so
randomness cannot reduce the attainable final count. The result is
`min(target, available nonredundant tagged markets)` per category.

Discovery strategy is part of the immutable cohort configuration. If a cohort was
started by an older release using the unfiltered global Gamma scan, start this
version with a new `--cohort` name; it will refuse to mix the two strategies.

Selected markets collect all API-served qualifying trades through the cohort's
frozen creation time, rather than only the latest 90 days. Trade workers run while
market discovery continues. Admission, semantic decisions, embeddings, cursors,
and chosen markets are durable and reproducible. Markets with no qualifying
trades remain part of the cohort.

Embedding device `auto` tries NVIDIA CUDA, Apple Metal (`mps`, including M-series
Macs), and CPU in that order. Accelerator failures fall back to the next device;
an explicit `cuda`, `mps`, or `cpu` selection is strict. Title vectors are cached
in SQLite, so a resumed run does not need to recompute them. The local model is
downloaded on its first use.

Use `report --cohort balanced-lifetime` to inspect discovery progress, random
rejections, eligible, selected, redundant, fetched, failed, zero-trade, trade,
event, and wallet counts by category. Passing
the same cohort to `export` additionally writes `cohort_markets.csv`,
`cohort_trades.csv`, `cohort_similarity_rejections.csv`,
`cohort_admission_rejections.csv`,
`cohort_resolutions.csv`, `cohort_labels.csv`, and
`cohort_label_fill_scenarios.csv`. Existing trades are not deleted and remain
visible in the global report and exports.

## Automated resolved-market pipeline

`run` is the default historical research workflow. It freezes the cohort cutoff,
streams closed markets, verifies terminal resolutions before embedding or
selection, selects the same balanced nonredundant lifetime sample, scans its trades,
enriches the configured subset, and labels every qualifying cohort trade:

```shell
python -m informed_flow --db data/informed_flow.sqlite3 run \
  --cohort balanced-resolved-lifetime
```

Existing cohort names retain their original configuration. A cohort created by
`sampled-backfill` cannot be silently converted into a resolution-required
cohort; choose a new name. Missing, pending, canceled, ambiguous, and otherwise
nonterminal resolutions are excluded before semantic selection. Finally settled
disputed markets remain eligible, with their raw resolution evidence retained.
Previously created bounded cohorts remain resumable with their original `--days`
value; omit `--days` when creating a new lifetime cohort.

The workflow uses a durable SQLite queue with unique jobs, renewable leases,
eight attempts with exponential backoff, crash recovery, and two workers each
for resolution verification, market scans, wallet enrichment, and labels. New
trades and their downstream jobs are committed atomically. On restart, missing
jobs are also reconstructed from cohort, trade, enrichment, and label state.
The command exits only after required historical jobs are terminal and returns
nonzero when a required job is dead. Inspect queue and label coverage with
`report --cohort balanced-resolved-90d`; after correcting a persistent failure,
use `retry-jobs` to return dead work to the queue.

Historical labels use +15 minute, +1 hour, and +24 hour targets. They take the
earliest price-history observation at or after the target, never a pre-target
point, and allow at most five minutes of delay. A public trade print in that
same interval is the explicitly approximate fallback. Daily token price series
are cached per cohort for reuse. If the market resolved before a target, that
horizon is recorded as unavailable. Historical prices are approximate and do
not claim executable order-book depth. Gross per-share P&L is stored; historical
fees and net P&L remain unknown unless a time-applicable fee source is later
verified.

Live polling is opt-in:

```shell
python -m informed_flow --db data/informed_flow.sqlite3 run \
  --cohort balanced-resolved-lifetime --live
```

Live delayed observations additionally retain top-of-book price, spread, depth,
and average-fill scenarios for $100, $500, and $1,000. The system remains
read-only and never submits an order.

The legacy unstratified `backfill --days` command discovers open and closed markets
through Gamma keyset pagination, then fetches each market's Data API v2 trade
history and enforces timestamps locally. The global trade feed ignores date bounds
and cannot serve a 90-day backfill. Coverage is limited to API-listed markets and
available history, not a guarantee of every historical trade. This legacy command
scans all listed markets and can take a long time. Use
`backfill --days 90 --max-markets 20` for a partial smoke test; the limit counts
discovered markets, including excluded markets and empty markets.

The date window is frozen on first run. Discovery and per-market page checkpoints
resume interrupted scans; duplicate trades are ignored. A completed scan retains
its original window. Use a new database path to start a fresh window. Old global
backfill checkpoints are ignored by the corrected implementation. For a clean
test, use a new database (e.g. `--db data/test_90d_v2.sqlite3`).

Backfill reuses metadata from discovery. Separate condition-ID metadata lookups
explicitly search both open and closed markets because Gamma defaults to open
markets. If processing a trade fails, its page is retained for retry and the
underlying error is printed before the scan stops.

Collection prints page progress and stores selected trades as pending enrichment
without downloading wallet histories. Run `enrich --limit 20` separately to try a
small batch, then `enrich` for the remaining selected trades. Wallet requests use
the 30 days immediately before the observed trade and print progress. The API
window is also enforced locally. A 100-page cap produces usable sample features
marked `history_truncated_at_100_pages`; repeated cursors and API errors still
fail enrichment. Wallet age and lifetime first activity stay null. The existing
`prior_trade_count` and `prior_market_count` fields now mean observed counts in
that recent window, not lifetime counts. Mean, median, and maximum size likewise
describe the observed window/sample. A truncated or malformed history receives
no low-activity points. Coverage and window bounds are preserved in wallet
`missing_reason` and exported as `wallet_coverage_details`.

Feature and score version 2 distinguish these semantics from the old lifetime
attempt. `enrich` automatically retries pending/failed trades and updates selected
trades completed under older feature versions. An enrichment marked complete
means processing succeeded; the wallet data may still be partial. Resolved
performance is calculated by the separate automated labeling jobs, never as a
wallet-enrichment feature. Current-window inactivity does not establish that a
wallet is new, and young-wallet cluster features remain unavailable.

Enrichment retains compact detailed summaries rather than raw wallet-history rows.
The summary includes observed trade and market counts, distinct outcomes, active
days, buy/sell counts and volume, total volume, size percentiles and standard
deviation, average trade price, recent seven-day activity, time since the last
observed trade, and market concentration with the top five markets. It covers
only the requested 30-day window/sample; lifetime age and totals remain unknown.
`wallet_history_fetches` retains status and window metadata, while
`wallet_history_summaries` stores the summary JSON. Summaries are reused for the
same wallet/window and exported to `wallet_summaries.csv`.

To replace previously collected raw enrichment history with these summaries and
reclaim disk space, stop enrichment and run `compact-wallet-history` on the same
database. It summarizes each stored fetch before deleting only
`wallet_history_raw` rows and vacuuming SQLite. Research trades, market metadata,
wallet features, scores, and checkpoints are retained. Raw research trade JSON
is still stored. Subsequent enrichment does not retain raw wallet-history rows.
New summary fields can require re-fetching history if they cannot be derived
from existing summaries. `report` shows both raw row and detailed summary counts.

The watch command overlaps each polling window by one minute. The overlap avoids
boundary gaps; duplicate rows are ignored. Ctrl+C stops after the active request
and transaction complete.

## Collection rules

- Store public taker `BUY` trades with notional value of at least $100.
- Include every market category except five-minute crypto Up/Down markets.
- Enrich every trade worth at least $500 and a deterministic 25% sample below it.
- Snapshot the live book for trades worth at least $1,000 or scoring at least
  medium.
- Never invent historical books; backfilled trades are marked
  `historical_unavailable`.
- Preserve the complete API payload for every accepted trade and market snapshot.

If Gamma metadata is unavailable, the trade is not admitted until market
classification can succeed. This prevents an ambiguous five-minute market from
entering the research dataset.

## Data and scoring

SQLite is authoritative. Currency, prices, and share quantities are stored in
millionths. Timestamps are UTC epoch seconds. `v_model_features` exposes the
latest feature and score versions for CSV export.

Wallet histories are requested with an exclusive cutoff before the observed
trade. Future rows are also rejected locally. Phase 1 intentionally leaves
resolved-performance fields null when they cannot be reconstructed without
future leakage.

The version 2 heuristic score adds points for a young wallet, little prior activity, large
or anomalous size, a cheap outcome, near resolution, a higher-prior category,
and clusters of young wallets. Every component is retained as JSON. The score is
a research hypothesis, not a trading recommendation.

## Tests

Tests use local fixtures/fakes and do not make network requests:

```shell
PYTHONPATH=src python -m unittest discover -v
```

## Operational notes

Polymarket APIs are external services. Errors are recorded in
`collection_errors`; retryable entity work appears in `pending_work`. Run
`enrich` to retry incomplete enrichments. Use `report` to inspect freshness,
coverage, exclusions, errors, categories, and recent high scores.

CSV exports are UTF-8 and portable to the later Windows training environment.
Docker/service packaging is intentionally deferred; the SQLite schema and CLI
are designed to remain unchanged when that wrapper is added.
