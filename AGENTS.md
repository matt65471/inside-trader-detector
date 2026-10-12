# Agent guidance

Before planning or implementing work in this repository, read
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md). It records the agreed scope,
architecture, data sources, constraints, and phased plan for the informed-flow
detector. Before any Phase 2 dataset construction or modeling, also read
[`docs/PHASE_2_TRAINING.md`](docs/PHASE_2_TRAINING.md).

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

## Current implementation and user decisions (2026-10-08)

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
  has a durable cursor; one page per active feed is fetched concurrently in stable
  round-robin rounds, and overlapping results deduplicate by condition ID. A seeded
  50% admission decision is made as markets arrive until every category reaches its
  quota. Same-event and >=0.90 embedding matches are rejected. If the relevant tag
  feeds exhaust, random rejects are reconsidered so the final count is
  `min(limit, available nonredundant tagged markets)`. Both `sampled-backfill` and
  `run` require targeted Gamma resolution verification before embedding or selection.
  Selected markets collect all API-served qualifying trades through a frozen cohort
  cutoff. Discovery strategy and resolution source are immutable; legacy cohorts
  require a new cohort name such as `balanced-tags-v2`.
- Gamma `GET /markets?condition_ids=...&closed=true&limit=1` is authoritative for
  historical eligibility and winner mapping. Require one exact condition row,
  `closed=true`, aligned unique outcomes/prices/CLOB tokens, exactly one price 1 and
  all others 0, and `umaResolutionStatus=resolved` when that field is present. Store
  all token/outcome indexes first, raw evidence, normalized payouts, source
  `gamma_markets_v1`, and `closedTime` only as `resolved_at`. Data `/v2/resolutions`
  rows cannot satisfy this gate. The Data API remains in use for trades and wallet
  enrichment.
- `run --cohort NAME` scans newly selected markets concurrently with continued
  discovery, enriches the selected subset, and labels every qualifying cohort trade
  at +15 minutes, +1 hour, and +24 hours. Live polling remains opt-in through `--live`.
- `finish-existing --cohort NAME` permanently freezes the current selection and
  performs no further discovery, selection, or reclassification. It requires all
  selected markets to already have Gamma verification, refuses legacy/incompatible
  cohorts without mutation, resumes incomplete scans, and reconciles enrichment and
  historical label jobs.
- Exhausted HTTP 500 responses must not stop a historical cohort. During discovery,
  mark only the failing Gamma tag stream exhausted, retain its `last_error`, and
  continue other tag feeds. For queued historical resolution, scan, enrichment, and
  labeling work, retain the entity's failed/incomplete state and terminate that job
  as an audited skip so other jobs continue. Do not apply permanent-skip behavior to
  live polling, where it could silently disable the only poller.
- The SQLite queue uses unique job keys, leases, crash recovery, eight attempts,
  exponential backoff, and dead-letter reporting. Trade insertion atomically adds
  downstream work, while startup reconciliation repairs missing jobs from
  authoritative cohort/trade state. Resolution jobs contain one condition ID. Scan,
  enrichment, and label workers recheck selected plus Gamma-verified state before
  network work; stale jobs finish as audited skips.
- Historical labels use the first post-target price-history point within five
  minutes, with a public trade print as the approximate fallback. They never use a
  pre-target point. Gross per-share P&L is populated from verified outcome payouts;
  historical fees and net P&L remain null without a verified time-applicable source.
  Historical prices never imply order-book depth. Optional live labels retain book
  and $100/$500/$1,000 fill scenarios.
- The user's current phase naming treats the entire acquisition, verification,
  enrichment, and labeling system as **Phase 1: data pipeline**. That implementation
  is complete. The active dataset still has to finish its queued work before its
  final counts are recorded. **Phase 2** is supervised model training; it is not yet
  implemented. See [`docs/PHASE_2_TRAINING.md`](docs/PHASE_2_TRAINING.md) for the
  training handoff. Do not train a model to reproduce the heuristic score.
- The summary-only, tag-filtered streaming sampled-backfill, queue, Gamma resolution,
  and labeling implementation plus the read-only Phase 2 audit were verified with 82 local tests. Update the validation record
  when subsequent code changes introduce new checks.

## Phase 1 operational handoff

The active historical dataset is on the Windows collection machine:

- Checkout: `C:\Users\matt6\projects\inside-trader-detector`
- Database: `C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.sqlite3`
- Cohort: `balanced-tags-v2`
- Scheduled task: `InformedFlow-gamma-v2`
- Log: `C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.log`
- Intended interpreter: `C:\Users\matt6\projects\inside-trader-detector\.venv311\Scripts\python.exe`

Do not delete or replace this database. `balanced-tags-v2` is the first cohort that
uses the required `gamma_markets_v1` gate. The old `balanced-tags-v1` and earlier
test databases are audit/history only and are not training inputs.

The collection command used for this dataset was:

```powershell
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  run `
  --cohort balanced-tags-v2 `
  --markets-per-category 1000 `
  --admission-rate 0.50 `
  --seed 0 `
  --similarity-threshold 0.90 `
  --market-workers 2 `
  --enrichment-workers 2 `
  --resolution-workers 2 `
  --label-workers 2 `
  --embedding-device auto
```

Discovery can be stopped permanently while completing the frozen selection with:

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

`finish-existing` is idempotent but permanently freezes the cohort. It discovers no
new markets and does not reclassify the selection. It resumes remaining selected-
market trade scans, builds/reconciles enrichment and label jobs, and exits when the
queue has completed or dead-lettered all relevant work. It runs market scans,
enrichment, and labels concurrently; the defaults are two workers for each group.
Resolution workers are effectively idle because the command refuses to start unless
every selected market already has terminal `gamma_markets_v1` evidence. One
enrichment worker is the conservative Windows setting because two trades can request
the same uncached wallet/window concurrently; that race wastes a duplicate fetch even
though the unique job lease prevents the same job from being claimed twice.
When launched directly from an SSH PowerShell session, the process is tied to that
session. For unattended completion, replace the pipeline command in
`run_gamma_v2.cmd` with the `finish-existing` invocation above and then start the
existing scheduled task; Task Scheduler will continue after SSH disconnects.

Useful operating commands are:

```powershell
# Report
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  report --cohort balanced-tags-v2

# Follow the log; Ctrl+C stops only the viewer
Get-Content "data\gamma_verified_v2.log" -Tail 50 -Wait

# Scheduled-task state and last exit code
Get-ScheduledTask -TaskName "InformedFlow-gamma-v2" |
  Select-Object TaskName, State
Get-ScheduledTaskInfo -TaskName "InformedFlow-gamma-v2" |
  Select-Object LastRunTime, LastTaskResult

# Start or stop the scheduled process
Start-ScheduledTask -TaskName "InformedFlow-gamma-v2"
Stop-ScheduledTask -TaskName "InformedFlow-gamma-v2"

# Main database plus SQLite WAL/SHM disk usage
Get-ChildItem "data\gamma_verified_v2.sqlite3*" |
  Select-Object Name,@{Name="SizeMB";Expression={[math]::Round($_.Length / 1MB,2)}}

# Portable training exports after finish-existing completes
.\.venv311\Scripts\python.exe -m informed_flow `
  --db "data\gamma_verified_v2.sqlite3" `
  export --cohort balanced-tags-v2 --output "data\gamma_verified_v2_export"
```

The scheduled launcher `run_gamma_v2.cmd` must invoke the `.venv311` interpreter
above. Do not restore the failing Python 3.12 executable. For GPU discovery, the
launcher must not contain `set CUDA_VISIBLE_DEVICES=-1`. GPU acceleration affects
semantic selection during discovery; `finish-existing` is mainly API and SQLite
work. The environment check used before restarting was:

```powershell
.\.venv311\Scripts\python.exe -X faulthandler -c "import argparse, sqlite3, torch, sentence_transformers; print('Environment OK'); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Python 3.12 previously crashed during `import argparse` with Windows status
`0xC0000005`, before pipeline code ran. Python 3.11 passed its standard-library
startup but Windows Smart App Control initially blocked PyTorch's unsigned
`torch\lib\shm.dll` with WinError 4551 and Code Integrity event 3077. Treat those
as machine/runtime issues, not SQLite or duplicate-job failures. If the machine
really has intermittent RAM errors, do not trust long database-writing runs until
Windows Memory Diagnostic passes.

### Current data snapshot (2026-10-08, not final)

The database was approximately 315.06 MB when the following report excerpt was
recorded. These counts may grow until `finish-existing` completes:

| Category | Eligible | Selected | Selected with trades | Qualifying trades | Zero-trade selected | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 4,136 | 239 | 95 | 1,924 | 144 | 761 |
| crypto | 11 | 4 | 2 | 16 | 2 | 996 |
| weather | 0 | 0 | 0 | 0 | 0 | 1,000 |
| culture | 912 | 73 | 50 | 225 | 23 | 927 |
| finance | 1,488 | 175 | 150 | 5,255 | 25 | 825 |
| geopolitics | 1,100 | 169 | 159 | 5,918 | 10 | 831 |
| politics | 655 | 104 | 94 | 7,069 | 10 | 896 |

The planned first training scope is sports, finance, geopolitics, and politics:
687 selected markets, 498 markets with at least one qualifying trade, and 20,166
trade rows in this snapshot. The trade count is not the independent sample size;
trades are clustered within markets, events, and wallets. Do not delete the other
categories; filter them at training time. Capture a new report and replace these
snapshot counts after finalization.

## Checkout and database locations

The collection runtime and authoritative active database are on Windows at the
paths in the Phase 1 handoff above. This documentation was updated from the Mac
checkout `/Users/matthewhu/Projects/inside-trader-detector`; edits there do not
automatically change the Windows checkout. Check the active checkout and Git status
before edits or commits, and sync changes deliberately. Do not claim changes are
committed, merged, pushed, or present on Windows unless verified.

An older collected test database may remain at
`C:\Users\matt6\.codex\worktrees\d527\inside-trader-detector\data\test_90d_v2.sqlite3`.
It is not the active Phase 2 source. Never delete raw research trades during
enrichment-history cleanup.

## Next steps

1. Run `finish-existing` for `balanced-tags-v2`, then save the final report and
   export. Inspect queue health, missing horizons, price-source quality, incomplete
   wallet windows, category balance, and concentration. A trade's enrichment status
   of `complete` means processing succeeded; its history can still be truncated.
2. Follow [`docs/PHASE_2_TRAINING.md`](docs/PHASE_2_TRAINING.md) to build the
   leakage-safe dataset audit and baseline evaluation. Do not treat the 20,166
   target-category trade rows as 20,166 independent examples.
3. Compare a simple supervised baseline against delayed market price and the
   existing heuristic. Train on older periods and test on newer markets/events.
   Report calibration and gross P&L by category; net P&L remains unknown until a
   historically applicable fee source is verified.
4. Improve status/reporting or collection only when the final audit identifies a
   concrete gap. Use a new cohort name if more discovery is needed because
   `balanced-tags-v2` is permanently frozen by `finish-existing`.

Do not start these larger new phases without user authorization. Keep execution,
wallet linking, and reinforcement learning outside the current scope.
