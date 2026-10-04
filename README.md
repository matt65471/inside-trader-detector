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
- No third-party runtime dependencies

Run directly from the checkout:

```shell
export PYTHONPATH=src
python -m informed_flow init
python -m informed_flow backfill --days 90
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
python -m informed_flow watch --once
python -m informed_flow watch --interval 60
python -m informed_flow enrich
python -m informed_flow rescore
python -m informed_flow report
python -m informed_flow export --output exports
```

Backfills use one-day windows, Data API v2 cursor pagination, and durable
checkpoints. Re-running a completed or interrupted window is safe: market IDs and
trade fingerprints are protected by database uniqueness constraints.

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

The version 1 score adds points for a young wallet, little prior activity, large
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

