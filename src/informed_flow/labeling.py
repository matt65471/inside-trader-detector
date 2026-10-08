"""Resolution verification and leakage-safe delayed-entry labels."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .api import APIError, PolymarketAPI
from .core import canonical_json, first, parse_timestamp, to_millionths
from .db import Database
from .service import parse_book

HORIZONS = (900, 3600, 86400)
PRICE_TOLERANCE_SECONDS = 300
GAMMA_RESOLUTION_SOURCE = "gamma_markets_v1"
_PRICE_LOCK_GUARD = threading.Lock()
_PRICE_LOCKS: dict[tuple[str, str, str, int], threading.Lock] = {}


def _array(value: Any) -> list[Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return list(value) if isinstance(value, list) else None


@dataclass(frozen=True)
class GammaResolutionEvidence:
    terminal: bool
    winner: int | None
    status: str
    quality: str
    outcomes: tuple[str, ...] = ()
    prices: tuple[Decimal, ...] = ()
    tokens: tuple[str, ...] = ()


def gamma_resolution_evidence(
    raw: Mapping[str, Any], expected_condition: str | None = None,
) -> GammaResolutionEvidence:
    condition = str(first(raw, "conditionId", "condition_id", default="")).strip().lower()
    if expected_condition is not None and condition != expected_condition.strip().lower():
        return GammaResolutionEvidence(False, None, "missing", "condition_mismatch")

    uma_value = first(raw, "umaResolutionStatus", "uma_resolution_status")
    uma_status = str(uma_value).strip().lower() if uma_value not in (None, "") else None
    status = uma_status or ("closed" if raw.get("closed") is True else "open")
    if raw.get("closed") is not True:
        return GammaResolutionEvidence(False, None, status, "not_closed")

    outcome_values = _array(first(raw, "outcomes"))
    price_values = _array(first(raw, "outcomePrices", "outcome_prices"))
    token_values = _array(first(raw, "clobTokenIds", "clob_token_ids"))
    if outcome_values is None:
        return GammaResolutionEvidence(False, None, status, "missing_outcomes")
    if price_values is None:
        return GammaResolutionEvidence(False, None, status, "missing_outcome_prices")
    if token_values is None:
        return GammaResolutionEvidence(False, None, status, "missing_clob_tokens")

    if any(not isinstance(value, str) for value in outcome_values):
        return GammaResolutionEvidence(False, None, status, "invalid_outcomes")
    if any(not isinstance(value, str) for value in token_values):
        return GammaResolutionEvidence(False, None, status, "invalid_clob_tokens")
    outcomes = tuple(value.strip() for value in outcome_values)
    tokens = tuple(value.strip() for value in token_values)
    try:
        prices = tuple(Decimal(str(value)) for value in price_values)
    except (InvalidOperation, ValueError):
        return GammaResolutionEvidence(False, None, status, "invalid_outcome_prices")
    if any(not price.is_finite() for price in prices):
        return GammaResolutionEvidence(False, None, status, "invalid_outcome_prices")

    if not outcomes or len(outcomes) != len(prices) or len(outcomes) != len(tokens):
        return GammaResolutionEvidence(False, None, status, "outcome_length_mismatch")
    if any(not value for value in outcomes):
        return GammaResolutionEvidence(False, None, status, "empty_outcome")
    if any(not value for value in tokens):
        return GammaResolutionEvidence(False, None, status, "empty_clob_token")
    if len({value.casefold() for value in outcomes}) != len(outcomes):
        return GammaResolutionEvidence(False, None, status, "duplicate_outcomes")
    if len(set(tokens)) != len(tokens):
        return GammaResolutionEvidence(False, None, status, "duplicate_clob_tokens")

    winners = [index for index, price in enumerate(prices) if price == Decimal(1)]
    if len(winners) != 1 or any(price not in (Decimal(0), Decimal(1)) for price in prices):
        return GammaResolutionEvidence(
            False, None, status, "ambiguous_outcome_prices", outcomes, prices, tokens,
        )
    if uma_status is not None and uma_status != "resolved":
        return GammaResolutionEvidence(
            False, None, uma_status, "nonterminal_resolution", outcomes, prices, tokens,
        )
    return GammaResolutionEvidence(
        True, winners[0], uma_status or "resolved", "terminal_outcome_prices",
        outcomes, prices, tokens,
    )


def resolution_result(raw: Mapping[str, Any]) -> tuple[bool, int | None, str, str]:
    """Compatibility wrapper exposing the normalized Gamma resolution decision."""
    evidence = gamma_resolution_evidence(raw)
    return evidence.terminal, evidence.winner, evidence.status, evidence.quality


def _explicit_dispute(raw: Mapping[str, Any]) -> bool:
    for key in ("wasDisputed", "was_disputed", "disputed", "umaDisputed"):
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    return False


class ResolutionStore:
    def __init__(self, db: Database, api: PolymarketAPI, *, clock=time.time):
        self.db = db
        self.api = api
        self.clock = clock

    def verify(self, cohort: str, conditions: list[str]) -> None:
        now = int(self.clock())
        for condition in conditions:
            rows = self.api.resolved_markets(condition)
            raw: Mapping[str, Any] = rows[0] if len(rows) == 1 else {}
            if not rows:
                evidence = GammaResolutionEvidence(False, None, "missing", "missing_resolution")
            elif len(rows) != 1:
                evidence = GammaResolutionEvidence(False, None, "ambiguous", "ambiguous_response")
            else:
                evidence = gamma_resolution_evidence(raw, condition)

            if evidence.terminal:
                placeholders = ",".join("?" for _ in evidence.tokens)
                conflicts = self.db.rows(
                    f"""SELECT asset_id,condition_id FROM outcomes
                         WHERE asset_id IN ({placeholders}) AND condition_id!=?""",
                    (*evidence.tokens, condition),
                )
                if conflicts:
                    evidence = replace(
                        evidence, terminal=False, winner=None, quality="token_condition_conflict",
                    )

            with self.db.connection:
                if evidence.terminal:
                    cohort_market = self.db.connection.execute(
                        """SELECT title,canonical_category,raw_json
                           FROM backfill_cohort_markets
                           WHERE cohort_name=? AND condition_id=?""",
                        (cohort, condition),
                    ).fetchone()
                    fallback_raw = json.loads(cohort_market["raw_json"]) if cohort_market else {}
                    event_rows = first(raw, "events", default=[]) or []
                    event = event_rows[0] if event_rows and isinstance(event_rows[0], Mapping) else {}
                    fee_schedule = first(raw, "feeSchedule", "fee_schedule", default={}) or {}
                    category = (
                        cohort_market["canonical_category"] if cohort_market else
                        first(event, "category", default=first(raw, "category"))
                    )
                    title = str(
                        first(raw, "question", "title", default=
                              cohort_market["title"] if cohort_market else condition)
                    )
                    slug = str(first(raw, "slug", default=first(fallback_raw, "slug", default=condition)))
                    fee_rate = first(fee_schedule, "rate")
                    try:
                        fee_rate_ppm = to_millionths(fee_rate) if fee_rate is not None else None
                    except (InvalidOperation, TypeError, ValueError):
                        fee_rate_ppm = None
                    self.db.connection.execute(
                        """INSERT INTO markets(
                               condition_id,event_id,title,slug,event_slug,category,start_ts,end_ts,
                               is_sports,accepting_orders,closed,fees_enabled,fee_rate_ppm,
                               first_seen_at,last_refreshed_at
                           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(condition_id) DO UPDATE SET
                               event_id=COALESCE(excluded.event_id,markets.event_id),
                               title=excluded.title,slug=excluded.slug,
                               event_slug=COALESCE(excluded.event_slug,markets.event_slug),
                               category=COALESCE(excluded.category,markets.category),
                               start_ts=COALESCE(excluded.start_ts,markets.start_ts),
                               end_ts=COALESCE(excluded.end_ts,markets.end_ts),
                               is_sports=excluded.is_sports,
                               accepting_orders=excluded.accepting_orders,closed=excluded.closed,
                               fees_enabled=excluded.fees_enabled,
                               fee_rate_ppm=COALESCE(excluded.fee_rate_ppm,markets.fee_rate_ppm),
                               last_refreshed_at=excluded.last_refreshed_at""",
                        (
                            condition, str(first(event, "id", default="")) or None,
                            title, slug, str(first(event, "slug", default="")) or None,
                            str(category).strip().lower() if category else None,
                            parse_timestamp(first(raw, "startDate", "start_date")),
                            parse_timestamp(first(raw, "endDate", "end_date", "endDateIso")),
                            int(bool(first(raw, "sportsMarketType", "sports_market_type", "gameStartTime"))),
                            int(bool(first(raw, "acceptingOrders", "accepting_orders", default=False))),
                            1, int(bool(first(raw, "feesEnabled", "fees_enabled", default=False))),
                            fee_rate_ppm, now, now,
                        ),
                    )
                    self.db.connection.executemany(
                        """INSERT INTO outcomes(
                               asset_id,condition_id,outcome_name,outcome_index,first_seen_at
                           ) VALUES (?,?,?,?,?)
                           ON CONFLICT(asset_id) DO UPDATE SET
                               outcome_name=excluded.outcome_name,
                               outcome_index=excluded.outcome_index""",
                        [
                            (token, condition, outcome, index, now)
                            for index, (token, outcome) in enumerate(
                                zip(evidence.tokens, evidence.outcomes)
                            )
                        ],
                    )
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
                        condition, evidence.status, int(evidence.terminal), evidence.winner,
                        canonical_json([str(value) for value in evidence.prices])
                        if evidence.prices else None,
                        parse_timestamp(first(raw, "closedTime", "closed_time")),
                        int(_explicit_dispute(raw)), GAMMA_RESOLUTION_SOURCE,
                        evidence.quality, canonical_json(rows), now,
                    ),
                )
                if not evidence.terminal:
                    self.db.connection.execute(
                        """UPDATE backfill_cohort_markets SET eligible=0,
                                  eligibility_reason=?,selection_status='ineligible',
                                  fetch_status='not_selected'
                           WHERE cohort_name=? AND condition_id=?""",
                        (evidence.quality, cohort, condition),
                    )


class HistoricalLabeler:
    def __init__(self, db: Database, api: PolymarketAPI, *, clock=time.time):
        self.db = db
        self.api = api
        self.clock = clock

    def _resolution(self, condition: str) -> Mapping[str, Any]:
        row = self.db.row("SELECT * FROM market_resolutions WHERE condition_id=?", (condition,))
        if not row or not row["terminal"] or row["source"] != GAMMA_RESOLUTION_SOURCE:
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
            """SELECT * FROM market_resolutions
               WHERE condition_id=? AND terminal=1 AND source=?""",
            (str(trade["condition_id"]), GAMMA_RESOLUTION_SOURCE),
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
