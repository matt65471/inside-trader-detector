from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 2

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS market_exclusions (
    condition_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    evidence_code TEXT NOT NULL,
    classified_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS markets (
    condition_id TEXT PRIMARY KEY,
    event_id TEXT,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    event_slug TEXT,
    category TEXT,
    start_ts INTEGER,
    end_ts INTEGER,
    is_sports INTEGER NOT NULL DEFAULT 0 CHECK (is_sports IN (0, 1)),
    accepting_orders INTEGER NOT NULL DEFAULT 0 CHECK (accepting_orders IN (0, 1)),
    closed INTEGER NOT NULL DEFAULT 0 CHECK (closed IN (0, 1)),
    fees_enabled INTEGER NOT NULL DEFAULT 0 CHECK (fees_enabled IN (0, 1)),
    fee_rate_ppm INTEGER,
    first_seen_at INTEGER NOT NULL,
    last_refreshed_at INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS reject_excluded_market
BEFORE INSERT ON markets
WHEN EXISTS (SELECT 1 FROM market_exclusions e WHERE e.condition_id = NEW.condition_id)
BEGIN
    SELECT RAISE(ABORT, 'excluded market');
END;

CREATE TABLE IF NOT EXISTS outcomes (
    asset_id TEXT PRIMARY KEY,
    condition_id TEXT NOT NULL REFERENCES markets(condition_id),
    outcome_name TEXT NOT NULL,
    outcome_index INTEGER,
    first_seen_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_market_name ON outcomes(condition_id, outcome_name);

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL REFERENCES markets(condition_id),
    observed_at INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    liquidity_microusd INTEGER,
    volume_24h_microusd INTEGER,
    best_bid_ppm INTEGER,
    best_ask_ppm INTEGER,
    last_price_ppm INTEGER,
    one_hour_change_ppm INTEGER,
    one_day_change_ppm INTEGER,
    raw_json TEXT NOT NULL,
    UNIQUE(condition_id, payload_hash)
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_time ON market_snapshots(condition_id, observed_at);

CREATE TABLE IF NOT EXISTS trades (
    trade_key TEXT PRIMARY KEY,
    source_trade_id TEXT,
    fingerprint TEXT NOT NULL UNIQUE,
    transaction_hash TEXT NOT NULL,
    proxy_wallet TEXT NOT NULL,
    condition_id TEXT NOT NULL REFERENCES markets(condition_id),
    asset_id TEXT NOT NULL REFERENCES outcomes(asset_id),
    side TEXT NOT NULL CHECK (side = 'BUY'),
    price_ppm INTEGER NOT NULL CHECK (price_ppm BETWEEN 0 AND 1000000),
    size_microshares INTEGER NOT NULL CHECK (size_microshares >= 0),
    notional_microusd INTEGER NOT NULL CHECK (notional_microusd >= 0),
    trade_ts INTEGER NOT NULL,
    ingested_at INTEGER NOT NULL,
    source_mode TEXT NOT NULL CHECK (source_mode IN ('live', 'backfill')),
    raw_json TEXT NOT NULL,
    enrichment_status TEXT NOT NULL CHECK (
        enrichment_status IN ('not_selected', 'pending', 'complete', 'failed')
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_source_id
    ON trades(source_trade_id) WHERE source_trade_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(trade_ts);
CREATE INDEX IF NOT EXISTS idx_trades_wallet_ts ON trades(proxy_wallet, trade_ts);
CREATE INDEX IF NOT EXISTS idx_trades_market_asset_ts ON trades(condition_id, asset_id, trade_ts);
CREATE INDEX IF NOT EXISTS idx_trades_notional ON trades(notional_microusd);

CREATE TRIGGER IF NOT EXISTS reject_excluded_trade
BEFORE INSERT ON trades
WHEN EXISTS (SELECT 1 FROM market_exclusions e WHERE e.condition_id = NEW.condition_id)
BEGIN
    SELECT RAISE(ABORT, 'excluded market trade');
END;

CREATE TABLE IF NOT EXISTS wallet_snapshots (
    wallet_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proxy_wallet TEXT NOT NULL,
    as_of_ts INTEGER NOT NULL,
    feature_version INTEGER NOT NULL,
    first_activity_ts INTEGER,
    wallet_age_seconds INTEGER,
    prior_trade_count INTEGER NOT NULL,
    prior_market_count INTEGER NOT NULL,
    mean_notional_microusd INTEGER,
    median_notional_microusd INTEGER,
    max_notional_microusd INTEGER,
    prior_resolved_count INTEGER,
    prior_wins INTEGER,
    prior_pnl_microusd INTEGER,
    completeness TEXT NOT NULL CHECK (completeness IN ('complete', 'partial', 'unavailable')),
    missing_reason TEXT,
    computed_at INTEGER NOT NULL,
    UNIQUE(proxy_wallet, as_of_ts, feature_version)
);

CREATE TABLE IF NOT EXISTS wallet_history_fetches (
    fetch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proxy_wallet TEXT NOT NULL,
    window_start INTEGER NOT NULL,
    window_end INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('fetching','complete','truncated','failed')),
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_wallet_history_window
    ON wallet_history_fetches(proxy_wallet,window_start,window_end,status);

CREATE TABLE IF NOT EXISTS wallet_history_raw (
    fetch_id INTEGER NOT NULL REFERENCES wallet_history_fetches(fetch_id),
    row_number INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY(fetch_id,row_number)
);

CREATE TABLE IF NOT EXISTS trade_features (
    trade_key TEXT NOT NULL REFERENCES trades(trade_key),
    feature_version INTEGER NOT NULL,
    wallet_snapshot_id INTEGER NOT NULL REFERENCES wallet_snapshots(wallet_snapshot_id),
    market_snapshot_id INTEGER REFERENCES market_snapshots(snapshot_id),
    size_vs_median_milli INTEGER,
    seconds_to_resolution INTEGER,
    outcome_price_ppm INTEGER NOT NULL,
    same_asset_buys_1h INTEGER NOT NULL,
    same_asset_notional_1h INTEGER NOT NULL,
    young_wallet_buys_1h INTEGER NOT NULL,
    category_prior TEXT NOT NULL CHECK (category_prior IN ('high', 'medium', 'low')),
    computed_at INTEGER NOT NULL,
    missing_fields_json TEXT NOT NULL,
    PRIMARY KEY(trade_key, feature_version)
);

CREATE TABLE IF NOT EXISTS scores (
    trade_key TEXT NOT NULL REFERENCES trades(trade_key),
    score_version INTEGER NOT NULL,
    score INTEGER NOT NULL,
    bucket TEXT NOT NULL CHECK (bucket IN ('low', 'medium', 'high')),
    components_json TEXT NOT NULL,
    computed_at INTEGER NOT NULL,
    PRIMARY KEY(trade_key, score_version)
);
CREATE INDEX IF NOT EXISTS idx_scores_bucket_score ON scores(bucket, score);

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    book_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_key TEXT NOT NULL REFERENCES trades(trade_key),
    asset_id TEXT NOT NULL REFERENCES outcomes(asset_id),
    snapshot_kind TEXT NOT NULL,
    requested_at INTEGER NOT NULL,
    observed_at INTEGER,
    latency_ms INTEGER,
    status TEXT NOT NULL CHECK (
        status IN ('complete', 'unavailable', 'historical_unavailable', 'failed')
    ),
    best_bid_ppm INTEGER,
    best_ask_ppm INTEGER,
    spread_ppm INTEGER,
    ask_depth_1c_microshares INTEGER,
    ask_depth_5c_microshares INTEGER,
    avg_fill_100_ppm INTEGER,
    avg_fill_500_ppm INTEGER,
    avg_fill_1000_ppm INTEGER,
    raw_json TEXT,
    error_message TEXT,
    UNIQUE(trade_key, snapshot_kind)
);

CREATE TABLE IF NOT EXISTS order_book_levels (
    book_snapshot_id INTEGER NOT NULL REFERENCES order_book_snapshots(book_snapshot_id) ON DELETE CASCADE,
    side TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
    level_number INTEGER NOT NULL,
    price_ppm INTEGER NOT NULL,
    size_microshares INTEGER NOT NULL,
    PRIMARY KEY(book_snapshot_id, side, level_number)
);

CREATE TABLE IF NOT EXISTS labels (
    trade_key TEXT NOT NULL REFERENCES trades(trade_key),
    horizon_seconds INTEGER NOT NULL,
    scheduled_at INTEGER NOT NULL,
    observed_at INTEGER,
    entry_price_ppm INTEGER,
    price_quality TEXT,
    outcome_won INTEGER,
    fee_microusd INTEGER,
    pnl_microusd INTEGER,
    PRIMARY KEY(trade_key, horizon_seconds)
);

CREATE TABLE IF NOT EXISTS collector_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    ended_at INTEGER,
    status TEXT NOT NULL,
    pages_fetched INTEGER NOT NULL DEFAULT 0,
    trades_seen INTEGER NOT NULL DEFAULT 0,
    trades_inserted INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    excluded_updown_5m INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    name TEXT PRIMARY KEY,
    cursor TEXT,
    window_start INTEGER,
    window_end INTEGER,
    last_trade_ts INTEGER,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_work (
    work_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER,
    last_error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(work_kind, entity_id)
);

CREATE TABLE IF NOT EXISTS collection_errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES collector_runs(run_id),
    stage TEXT NOT NULL,
    endpoint TEXT,
    entity_type TEXT,
    entity_id TEXT,
    occurred_at INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
    status_code INTEGER,
    error_message TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS v_model_features AS
WITH latest_features AS (
    SELECT tf.* FROM trade_features tf
    JOIN (
        SELECT trade_key, MAX(feature_version) AS version
        FROM trade_features GROUP BY trade_key
    ) latest ON latest.trade_key = tf.trade_key AND latest.version = tf.feature_version
), latest_scores AS (
    SELECT s.* FROM scores s
    JOIN (
        SELECT trade_key, MAX(score_version) AS version
        FROM scores GROUP BY trade_key
    ) latest ON latest.trade_key = s.trade_key AND latest.version = s.score_version
)
SELECT
    t.trade_key, t.trade_ts, t.proxy_wallet, t.condition_id, t.asset_id,
    t.price_ppm, t.size_microshares, t.notional_microusd,
    m.title, m.slug, m.event_slug, m.category, m.is_sports, m.end_ts,
    o.outcome_name,
    ws.wallet_age_seconds, ws.prior_trade_count, ws.prior_market_count,
    ws.mean_notional_microusd, ws.median_notional_microusd,
    ws.max_notional_microusd, ws.prior_resolved_count, ws.prior_wins,
    ws.prior_pnl_microusd, ws.completeness AS wallet_completeness,
    ws.missing_reason AS wallet_coverage_details,
    lf.size_vs_median_milli, lf.seconds_to_resolution,
    lf.same_asset_buys_1h, lf.same_asset_notional_1h,
    lf.young_wallet_buys_1h, lf.category_prior, lf.missing_fields_json,
    ls.score, ls.bucket, ls.components_json,
    ob.best_bid_ppm, ob.best_ask_ppm, ob.spread_ppm,
    ob.ask_depth_1c_microshares, ob.ask_depth_5c_microshares,
    ob.avg_fill_100_ppm, ob.avg_fill_500_ppm, ob.avg_fill_1000_ppm,
    ob.status AS book_status
FROM trades t
JOIN markets m ON m.condition_id = t.condition_id
JOIN outcomes o ON o.asset_id = t.asset_id
JOIN latest_features lf ON lf.trade_key = t.trade_key
JOIN wallet_snapshots ws ON ws.wallet_snapshot_id = lf.wallet_snapshot_id
LEFT JOIN latest_scores ls ON ls.trade_key = t.trade_key
LEFT JOIN order_book_snapshots ob
    ON ob.trade_key = t.trade_key AND ob.snapshot_kind = 'post_detection';
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        with self.connection:
            self.connection.executescript(SCHEMA_SQL)
            # Refresh the export view when upgrading an existing database.
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(v_model_features)")}
            if "wallet_coverage_details" not in columns:
                self.connection.execute("DROP VIEW v_model_features")
                self.connection.executescript(SCHEMA_SQL)
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, int(time.time())),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def row(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, params))

    def exclude_market(self, condition_id: str, evidence: str, now: int) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO market_exclusions(condition_id, reason, evidence_code, classified_at)
                   VALUES (?, 'crypto_updown_5m', ?, ?)
                   ON CONFLICT(condition_id) DO UPDATE SET
                       evidence_code=excluded.evidence_code,
                       classified_at=excluded.classified_at""",
                (condition_id, evidence, now),
            )

    def is_excluded(self, condition_id: str) -> bool:
        return self.row(
            "SELECT 1 FROM market_exclusions WHERE condition_id = ?", (condition_id,)
        ) is not None

    def start_run(self, mode: str, now: int) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO collector_runs(mode, started_at, status) VALUES (?, ?, 'running')",
                (mode, now),
            )
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, counts: Mapping[str, int], now: int) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE collector_runs SET ended_at=?, status=?, pages_fetched=?, trades_seen=?,
                       trades_inserted=?, duplicates=?, excluded_updown_5m=?, errors=?
                   WHERE run_id=?""",
                (
                    now,
                    status,
                    counts.get("pages", 0),
                    counts.get("seen", 0),
                    counts.get("inserted", 0),
                    counts.get("duplicates", 0),
                    counts.get("excluded", 0),
                    counts.get("errors", 0),
                    run_id,
                ),
            )

    def record_error(
        self,
        run_id: int | None,
        stage: str,
        message: str,
        *,
        endpoint: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO collection_errors(
                       run_id, stage, endpoint, entity_type, entity_id, occurred_at,
                       attempts, retryable, status_code, error_message
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, stage, endpoint, entity_type, entity_id, int(time.time()), attempts,
                    int(retryable), status_code, message,
                ),
            )

    def checkpoint(self, name: str) -> sqlite3.Row | None:
        return self.row("SELECT * FROM checkpoints WHERE name = ?", (name,))

    def save_checkpoint(
        self,
        name: str,
        *,
        cursor: str | None,
        window_start: int | None,
        window_end: int | None,
        last_trade_ts: int | None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO checkpoints(name, cursor, window_start, window_end, last_trade_ts, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET cursor=excluded.cursor,
                       window_start=excluded.window_start, window_end=excluded.window_end,
                       last_trade_ts=excluded.last_trade_ts, updated_at=excluded.updated_at""",
                (name, cursor, window_start, window_end, last_trade_ts, int(time.time())),
            )

    def upsert_pending(self, kind: str, entity_id: str, error: str, now: int) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO pending_work(
                       work_kind, entity_id, status, attempt_count, next_attempt_at,
                       last_error, created_at, updated_at
                   ) VALUES (?, ?, 'pending', 1, ?, ?, ?, ?)
                   ON CONFLICT(work_kind, entity_id) DO UPDATE SET
                       status='pending', attempt_count=pending_work.attempt_count+1,
                       next_attempt_at=excluded.next_attempt_at,
                       last_error=excluded.last_error, updated_at=excluded.updated_at""",
                (kind, entity_id, now + 60, error, now, now),
            )

    def resolve_pending(self, kind: str, entity_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM pending_work WHERE work_kind=? AND entity_id=?", (kind, entity_id)
            )

    def dump_json(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

