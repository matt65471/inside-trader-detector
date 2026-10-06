"""Resolution verification and leakage-safe delayed-entry labels."""
from __future__ import annotations

import json
import threading
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .api import APIError, PolymarketAPI
from .core import canonical_json, first, parse_timestamp, to_millionths
from .db import Database
from .service import parse_book

HORIZONS = (900, 3600, 86400)
PRICE_TOLERANCE_SECONDS = 300
TERMINAL_STATUSES = {"resolved", "finalized", "settled", "complete", "completed"}
_PRICE_LOCK_GUARD = threading.Lock()
_PRICE_LOCKS: dict[tuple[str, str, str, int], threading.Lock] = {}


def _payouts(raw: Mapping[str, Any]) -> list[Decimal]:
    values = raw.get("payouts")
    if not isinstance(values, list):
        return []
    try:
        return [Decimal(str(value)) for value in values]
    except (InvalidOperation, ValueError):
        return []


def resolution_result(raw: Mapping[str, Any]) -> tuple[bool, int | None, str, str]:
    payouts = _payouts(raw)
    scale = max(payouts, default=Decimal(0))
    # Data API v2 serves micro-USDC payouts; accept the older normalized fixture
    # form as well so stored evidence from either API generation remains usable.
    binary = scale in (Decimal(1), Decimal(1_000_000)) and all(
        value in (0, scale) for value in payouts
    )
    winners = [index for index, value in enumerate(payouts) if value == scale]
    binary = binary and len(winners) == 1
    status = str(raw.get("status") or "unknown").lower()
    terminal = binary and status in TERMINAL_STATUSES
    if terminal:
        return True, winners[0], status, "terminal_payout"
    if not payouts:
        return False, None, status, "missing_payouts"
    if not binary:
        return False, None, status, "ambiguous_payouts"
    return False, None, status, "nonterminal_resolution"


class ResolutionStore:
    def __init__(self, db: Database, api: PolymarketAPI, *, clock=time.time):
        self.db = db
        self.api = api
        self.clock = clock

    def verify(self, cohort: str, conditions: list[str]) -> None:
        rows = self.api.resolutions(conditions)
        by_condition = {
            str(first(row, "condition_id", "conditionId", default="")).lower(): row
            for row in rows
        }
        now = int(self.clock())
        with self.db.connection:
            for condition in conditions:
                raw = by_condition.get(condition.lower())
                if raw is None:
                    terminal, winner, status, quality = False, None, "missing", "missing_resolution"
                    raw = {}
                else:
                    terminal, winner, status, quality = resolution_result(raw)
                self.db.connection.execute(
                    """INSERT INTO market_resolutions(
                           condition_id,status,terminal,winning_outcome_index,payouts_json,
                           resolved_at,was_disputed,source,quality,raw_json,checked_at,error_message
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)
                       ON CONFLICT(condition_id) DO UPDATE SET
                           status=excluded.status,terminal=excluded.terminal,
                           winning_outcome_index=excluded.winning_outcome_index,
                           payouts_json=excluded.payouts_json,resolved_at=excluded.resolved_at,
                           was_disputed=excluded.was_disputed,source=excluded.source,
                           quality=excluded.quality,raw_json=excluded.raw_json,
                           checked_at=excluded.checked_at,error_message=NULL""",
                    (
                        condition, status, int(terminal), winner,
                        canonical_json(raw.get("payouts")) if raw.get("payouts") is not None else None,
                        parse_timestamp(raw.get("resolved_at")), int(bool(raw.get("was_disputed"))),
                        "data_v2_resolutions", quality, canonical_json(raw), now,
                    ),
                )
                if not terminal:
                    self.db.connection.execute(
                        """UPDATE backfill_cohort_markets SET eligible=0,
                                  eligibility_reason=?,selection_status='ineligible',
                                  fetch_status='not_selected'
                           WHERE cohort_name=? AND condition_id=?""",
                        (quality, cohort, condition),
                    )


class HistoricalLabeler:
    def __init__(self, db: Database, api: PolymarketAPI, *, clock=time.time):
        self.db = db
        self.api = api
        self.clock = clock

    def _resolution(self, condition: str) -> Mapping[str, Any]:
        row = self.db.row("SELECT * FROM market_resolutions WHERE condition_id=?", (condition,))
        if not row or not row["terminal"]:
            raise APIError(f"No terminal resolution for {condition}", retryable=True)
        return row

    def _history(self, cohort: str, asset: str, start: int, end: int) -> list[dict[str, Any]]:
        # Cache one UTC day per token. The returned points are still filtered to
        # the exact post-target five-minute interval, so cache reuse cannot leak
        # a pre-target or far-future price into a label.
        bucket_start = max(1, (start // 86400) * 86400)
        bucket_end = bucket_start + 86400 + PRICE_TOLERANCE_SECONDS
        lock_key = (str(self.db.path), cohort, asset, bucket_start)
        with _PRICE_LOCK_GUARD:
            lock = _PRICE_LOCKS.setdefault(lock_key, threading.Lock())
        with lock:
            cached = self.db.row(
                """SELECT * FROM price_history_cache WHERE cohort_name=? AND asset_id=?
                   AND window_start=? AND window_end=? AND fidelity_minutes=1""",
                (cohort, asset, bucket_start, bucket_end),
            )
            if cached and cached["status"] == "complete":
                return json.loads(cached["raw_json"] or "[]")
            try:
                rows = self.api.price_history(asset, bucket_start, bucket_end, fidelity=1)
            except APIError as exc:
                with self.db.connection:
                    self.db.connection.execute(
                        """INSERT INTO price_history_cache(
                               cohort_name,asset_id,window_start,window_end,fidelity_minutes,
                               status,raw_json,fetched_at,error_message
                           ) VALUES (?,?,?,?,1,'failed',NULL,?,?)
                           ON CONFLICT(cohort_name,asset_id,window_start,window_end,fidelity_minutes)
                           DO UPDATE SET status='failed',fetched_at=excluded.fetched_at,
                                         error_message=excluded.error_message""",
                        (cohort, asset, bucket_start, bucket_end, int(self.clock()), str(exc)),
                    )
                raise
            with self.db.connection:
                self.db.connection.execute(
                    """INSERT INTO price_history_cache(
                           cohort_name,asset_id,window_start,window_end,fidelity_minutes,
                           status,raw_json,fetched_at,error_message
                       ) VALUES (?,?,?,?,1,'complete',?,?,NULL)
                       ON CONFLICT(cohort_name,asset_id,window_start,window_end,fidelity_minutes)
                       DO UPDATE SET status='complete',raw_json=excluded.raw_json,
                                     fetched_at=excluded.fetched_at,error_message=NULL""",
                    (
                        cohort, asset, bucket_start, bucket_end,
                        canonical_json(rows), int(self.clock()),
                    ),
                )
            return rows

    @staticmethod
    def _history_point(rows: list[Mapping[str, Any]], target: int) -> tuple[int, int] | None:
        choices: list[tuple[int, int]] = []
        for row in rows:
            try:
                timestamp = int(first(row, "t", "timestamp"))
                price = to_millionths(first(row, "p", "price"))
            except (TypeError, ValueError):
                continue
            if target <= timestamp <= target + PRICE_TOLERANCE_SECONDS:
                choices.append((timestamp, price))
        return min(choices) if choices else None

    def _trade_print(self, asset: str, target: int) -> tuple[int, int] | None:
        row = self.db.row(
            """SELECT trade_ts,price_ppm FROM trades WHERE asset_id=?
               AND trade_ts BETWEEN ? AND ? ORDER BY trade_ts LIMIT 1""",
            (asset, target, target + PRICE_TOLERANCE_SECONDS),
        )
        return (int(row["trade_ts"]), int(row["price_ppm"])) if row else None

    def _upsert_label(
        self,
        trade: Mapping[str, Any],
        horizon: int,
        *,
        status: str,
        observed_at: int | None = None,
        price: int | None = None,
        source: str | None = None,
        quality: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        book: Mapping[str, Any] | None = None,
    ) -> None:
        resolution = self.db.row(
            "SELECT * FROM market_resolutions WHERE condition_id=? AND terminal=1",
            (str(trade["condition_id"]),),
        )
        outcome_index = trade["outcome_index"]
        won = (
            None if outcome_index is None or resolution is None
            else int(int(outcome_index) == resolution["winning_outcome_index"])
        )
        gross = None
        if price is not None and won is not None:
            gross = 1_000_000 - price if won else -price
        target = int(trade["trade_ts"]) + horizon
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO labels(
                       trade_key,horizon_seconds,scheduled_at,observed_at,entry_price_ppm,
                       price_quality,outcome_won,fee_microusd,pnl_microusd,status,
                       price_source,gross_pnl_microusd,fee_source,net_pnl_microusd,source_json
                       ,best_bid_ppm,spread_ppm,ask_depth_1c_microshares,
                       ask_depth_5c_microshares
                   ) VALUES (?,?,?,?,?,?,?,NULL,NULL,?,?,?,NULL,NULL,?,?,?,?,?)
                   ON CONFLICT(trade_key,horizon_seconds) DO UPDATE SET
                       scheduled_at=excluded.scheduled_at,observed_at=excluded.observed_at,
                       entry_price_ppm=excluded.entry_price_ppm,
                       price_quality=excluded.price_quality,outcome_won=excluded.outcome_won,
                       status=excluded.status,price_source=excluded.price_source,
                       gross_pnl_microusd=excluded.gross_pnl_microusd,
                       source_json=excluded.source_json,
                       best_bid_ppm=excluded.best_bid_ppm,
                       spread_ppm=excluded.spread_ppm,
                       ask_depth_1c_microshares=excluded.ask_depth_1c_microshares,
                       ask_depth_5c_microshares=excluded.ask_depth_5c_microshares""",
                (
                    trade["trade_key"], horizon, target, observed_at, price, quality, won,
                    status, source, gross, canonical_json(evidence or {}),
                    (book or {}).get("best_bid_ppm"), (book or {}).get("spread_ppm"),
                    (book or {}).get("ask_depth_1c_microshares"),
                    (book or {}).get("ask_depth_5c_microshares"),
                ),
            )

    def observe(self, cohort: str, trade_key: str, horizon: int) -> None:
        trade = self.db.row(
            """SELECT t.*,o.outcome_index FROM trades t JOIN outcomes o USING(asset_id)
               WHERE t.trade_key=?""",
            (trade_key,),
        )
        if not trade:
            raise ValueError(f"Unknown trade: {trade_key}")
        target = int(trade["trade_ts"]) + horizon
        resolution = self._resolution(str(trade["condition_id"]))
        if resolution["resolved_at"] is not None and int(resolution["resolved_at"]) <= target:
            self._upsert_label(trade, horizon, status="unavailable_before_horizon")
            return
        rows = self._history(cohort, trade["asset_id"], target, target + PRICE_TOLERANCE_SECONDS)
        point = self._history_point(rows, target)
        source = "historical_price_history"
        if point is None:
            point = self._trade_print(trade["asset_id"], target)
            source = "historical_trade_print"
        if point is None:
            self._upsert_label(trade, horizon, status="price_unavailable")
            return
        observed, price = point
        self._upsert_label(
            trade, horizon, status="complete", observed_at=observed, price=price,
            source=source, quality="approximate", evidence={"timestamp": observed, "price_ppm": price},
        )

    def observe_live(self, trade_key: str, horizon: int) -> None:
        trade = self.db.row(
            """SELECT t.*,o.outcome_index FROM trades t JOIN outcomes o USING(asset_id)
               WHERE t.trade_key=?""", (trade_key,),
        )
        if not trade:
            raise ValueError(f"Unknown trade: {trade_key}")
        target = int(trade["trade_ts"]) + horizon
        now = int(self.clock())
        if now > target + PRICE_TOLERANCE_SECONDS:
            self.observe("live", trade_key, horizon)
            return
        raw = self.api.book(trade["asset_id"])
        parsed = parse_book(raw)
        price = parsed["best_ask_ppm"]
        if price is None:
            self._upsert_label(trade, horizon, status="price_unavailable")
            return
        self._upsert_label(
            trade, horizon, status="observed_pending_resolution", observed_at=now,
            price=price, source="live_order_book", quality="executable_top_of_book",
            evidence=raw, book=parsed,
        )
        for notional, key in ((100, "avg_fill_100_ppm"), (500, "avg_fill_500_ppm"), (1000, "avg_fill_1000_ppm")):
            average = parsed[key]
            shares = notional * 1_000_000_000_000 // average if average else None
            with self.db.connection:
                self.db.connection.execute(
                    """INSERT INTO label_fill_scenarios(
                           trade_key,horizon_seconds,target_notional_microusd,
                           average_fill_price_ppm,shares_microshares
                       ) VALUES (?,?,?,?,?)
                       ON CONFLICT(trade_key,horizon_seconds,target_notional_microusd)
                       DO UPDATE SET average_fill_price_ppm=excluded.average_fill_price_ppm,
                                     shares_microshares=excluded.shares_microshares""",
                    (trade_key, horizon, notional * 1_000_000, average, shares),
                )

    def finalize_market(self, condition_id: str) -> None:
        resolution = self._resolution(condition_id)
        rows = self.db.rows(
            """SELECT l.*,o.outcome_index FROM labels l
               JOIN trades t ON t.trade_key=l.trade_key
               JOIN outcomes o ON o.asset_id=t.asset_id
               WHERE t.condition_id=? AND l.entry_price_ppm IS NOT NULL""",
            (condition_id,),
        )
        with self.db.connection:
            for row in rows:
                won = int(row["outcome_index"] == resolution["winning_outcome_index"])
                price = int(row["entry_price_ppm"])
                gross = 1_000_000 - price if won else -price
                self.db.connection.execute(
                    """UPDATE labels SET outcome_won=?,gross_pnl_microusd=?,
                              status=CASE WHEN status='observed_pending_resolution'
                                          THEN 'complete' ELSE status END
                       WHERE trade_key=? AND horizon_seconds=?""",
                    (won, gross, row["trade_key"], row["horizon_seconds"]),
                )
                scenarios = self.db.rows(
                    """SELECT * FROM label_fill_scenarios
                       WHERE trade_key=? AND horizon_seconds=?""",
                    (row["trade_key"], row["horizon_seconds"]),
                )
                for scenario in scenarios:
                    cost = int(scenario["target_notional_microusd"])
                    shares = scenario["shares_microshares"]
                    scenario_gross = None
                    if shares is not None:
                        payout = int(shares) if won else 0
                        scenario_gross = payout - cost
                    self.db.connection.execute(
                        """UPDATE label_fill_scenarios SET gross_pnl_microusd=?
                           WHERE trade_key=? AND horizon_seconds=?
                             AND target_notional_microusd=?""",
                        (
                            scenario_gross, row["trade_key"], row["horizon_seconds"],
                            scenario["target_notional_microusd"],
                        ),
                    )
