# Agent guidance

Before planning or implementing work in this repository, read
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md). It records the agreed scope,
architecture, data sources, constraints, and phased plan for the informed-flow
detector.

The most important guardrails are:

- Keep the system read-only. Do not place trades or add execution code.
- Model whether a public trade appears informed and whether following it after a
  realistic delay would have been profitable; do not claim to identify legal
  insiders.
- Do not add wallet linking, reinforcement learning, or a trading bot unless the
  user explicitly expands the scope.
- Store broadly before scoring; scores must be recomputable.
- Ask before undertaking a large new build unless the user explicitly says to
  start implementation.

## Current implementation and user decisions (2026-10-06)

The implementation now exists in `src/informed_flow/`. Some implementation-status
statements in `docs/PROJECT_CONTEXT.md` describe an earlier checkout. Use this
section for the current status, while retaining that document's research scope.

- Phase 1 collects public taker BUY trades worth at least $100. Five-minute
  crypto Up/Down markets are excluded. Store raw research-trade JSON and raw
  market metadata; duplicate trade IDs/fingerprints are ignored.
- Historical backfill discovers closed markets first, then open markets, through
  Gamma keyset pagination, and fetches trades by condition ID. Market/global
  Data API feeds ignore date bounds, so enforce trade timestamps locally.
  Market discovery scans all listed markets; an unfinished newest-first scan can
  be heavily skewed toward sports. Coverage is limited to API-served history.
- A backfill freezes its requested date window and saves discovery/per-market
  page checkpoints. Resume with the same database and `--days` value. Failed
  processing pages remain pending for replay; already stored trades are retained.
  Output counters reset per market and `seen` counts rows after date filtering.
- Backfill reuses discovery metadata. Separate Gamma metadata lookups explicitly
  search both open and closed markets: the default `closed=false` previously
  caused valid closed-market trades to fail classification.
- Collection and enrichment are separate steps within Phase 1. Enrich every
  stored trade worth at least $500 plus a deterministic 25% sample below $500.
  `enrich` retries failed/pending work and refreshes outdated features.
- Wallet enrichment requests the 30 days BEFORE each observed trade (upper bound
  is trade timestamp minus one second), also enforcing timestamps locally.
  The 100-page cap yields a marked truncated sample rather than failing the
  entire enrichment. API failures/repeated cursors still fail enrichment.
- **Explicit user storage decision:** retain compiled, detailed wallet summaries,
  not individual raw wallet-history rows. This is an authorized exception to
  broadly retaining raw enrichment data; do not silently re-enable raw history
  retention. Raw research trades and market metadata are still retained.
- Summaries retain observed counts, distinct markets/outcomes, active days,
  buy/sell counts and volume, size mean/min/max/median/percentiles/deviation,
  average execution price, cheap-trade count, seven-day activity, recency,
  market concentration, and the top five markets. Save window bounds, fetch
  status, malformed/out-of-window counts, and truncation metadata separately.
  Counts and size statistics describe the observed recent window/sample, not
  lifetime activity. Wallet age, lifetime first activity, and prior resolved
  wins/P&L remain unknown. Do not infer a young wallet from recent history alone.
- `wallet_history_fetches` stores request/coverage metadata;
  `wallet_history_summaries` stores compact summary JSON. Reuse summaries for
  identical wallet/windows. Features can be recomputed from existing summaries
  only when those summaries retain the required information; new features may
  require refetching history. Summaries export as `wallet_summaries.csv`.
- `compact-wallet-history` summarizes previously stored enrichment history,
  deletes only `wallet_history_raw` rows, and vacuums SQLite. Stop enrichment
  before cleanup. The cleanup already preserved 19,840 research trades and all
  existing features/scores/checkpoints while removing 336,159 raw enrichment
  rows, reducing the test database from roughly 347 MB to 62 MB. These counts
  describe the cleanup, not permanent dataset totals.
- Version 2 features/scores keep the 30-day semantics separate from the original
  lifetime-history attempt. The heuristic scores individual trades using their
  wallet/market context. It is a baseline, not a calibrated probability or a
  training label. Truncated histories receive no low-activity-count points.
- Historical order books are not recoverable from these trades and are marked
  `historical_unavailable`. Live collection attempts book snapshots for trades
  worth at least $1,000. Completed-trade price is not a follower's executable ask.
- New `sampled-backfill` cohorts resolve Gamma tag slugs to numeric IDs and stream
  separate closed-market feeds for all seven categories and their aliases. Each tag
  has a durable cursor; overlapping results deduplicate by condition ID. A seeded
  50% admission decision is made as markets arrive until every category reaches its
  quota. Same-event and >=0.90 embedding matches are rejected. If the relevant tag
  feeds exhaust, random rejects are reconsidered so the final count is
  `min(limit, available nonredundant tagged markets)`. Selected markets collect all
  API-served qualifying trades through a frozen cohort cutoff. Discovery strategy
  is immutable; older global-scan cohorts require a new cohort name.
- `run --cohort NAME` automates resolved lifetime cohorts. It verifies terminal Data
  API resolutions before immediate embedding/selection, scans newly selected markets
  concurrently with continued discovery, enriches the selected subset, and labels
  every qualifying cohort trade at +15 minutes, +1 hour, and +24 hours. Live polling
  remains opt-in through `--live`.
- The SQLite queue uses unique job keys, leases, crash recovery, eight attempts,
  exponential backoff, and dead-letter reporting. Trade insertion atomically adds
  downstream work, while startup reconciliation repairs missing jobs from
  authoritative cohort/trade state.
- Historical labels use the first post-target price-history point within five
  minutes, with a public trade print as the approximate fallback. They never use a
  pre-target point. Gross per-share P&L is populated from verified outcome payouts;
  historical fees and net P&L remain null without a verified time-applicable source.
  Historical prices never imply order-book depth. Optional live labels retain book
  and $100/$500/$1,000 fill scenarios.
- Phase 2 historical labeling and gross-profit calculation are implemented. Model
  training, profitability claims, and paper trading are **not implemented**. Do not
  train a model to reproduce the heuristic score.
- The summary-only, tag-filtered streaming sampled-backfill, queue, resolution, and
  labeling implementation was verified with 71 local tests. Update the validation record
  when subsequent code changes introduce new checks.

## Checkout and database locations

The user runs VS Code from `C:\Users\matt6\projects\inside-trader-detector`.
This chat's workspace is
`C:\Users\matt6\.codex\worktrees\d527\inside-trader-detector`.
The latest summary-only changes were applied in both locations; do not assume
edits in one automatically update the other. Check the active checkout and Git
status before edits/commits and explain where changes live. Do not claim changes
are committed, merged, or pushed unless those actions were verified.

The collected test database remains at
`C:\Users\matt6\.codex\worktrees\d527\inside-trader-detector\data\test_90d_v2.sqlite3`.
Use that absolute `--db` path when running from the original checkout. A relative
database path there points to a different database. Never delete raw research
trades during enrichment-history cleanup.

## Next steps

1. Finish/resume historical collection and enrich the selected trades using the
   summary-only implementation. Inspect coverage, missing fields, errors, storage
   growth, score distributions, and category imbalance. A trade's enrichment
   status of `complete` means processing succeeded; history may remain partial.
2. Improve status reporting so users can distinguish API-returned rows, rows
   outside the window, and saved trades, and inspect backfill completion directly
   rather than treating an ingestion count as proof of completion.
3. Run and inspect the resolved historical pipeline: resolution exclusions, queue
   health, missing horizons, price-source quality, unknown fees, and gross P&L.
   Verify any fee schedule against a source applicable at the historical trade time
   before calculating net P&L.
4. Evaluate outcomes/profit by heuristic score and category, comparing with market
   price as the baseline. Account for category coverage, concentration in a few
   wallets/events, and longshot outliers before claiming an edge.
5. Only after reliable labels exist, train a simple supervised model from features
   and observed outcomes/profit. Train on older periods and evaluate on newer
   periods. Compare against the heuristic and market-price baselines; keep any
   later paper-action rule separate from model predictions.

Do not start these larger new phases without user authorization. Keep execution,
wallet linking, and reinforcement learning outside the current scope.
