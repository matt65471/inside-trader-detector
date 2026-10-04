from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .api import APIError, HistoryTruncated, Page, PolymarketAPI
from .core import (
    FEATURE_VERSION,
    SCORE_VERSION,
    canonical_json,
    category_prior,
    classify_five_minute_updown,
    deterministic_sample,
    first,
    median_int,
    normalize_trade,
    notional_microusd,
    parse_jsonish,
    parse_timestamp,
    payload_hash,
    to_millionths,
)
from .db import Database

MIN_NOTIONAL_MICROUSD = 100_000_000
ENRICH_NOTIONAL_MICROUSD = 500_000_000
BOOK_NOTIONAL_MICROUSD = 1_000_000_000
MARKET_CACHE_SECONDS = 6 * 60 * 60


@dataclass
class Counts:
    pages: int = 0
    seen: int = 0
    inserted: int = 0
    duplicates: int = 0
    excluded: int = 0
    errors: int = 0
    enriched: int = 0
    high_scores: list[tuple[str, int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "pages": self.pages,
            "seen": self.seen,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "excluded": self.excluded,
            "errors": self.errors,
        }


def _bool(value: Any) -> int:
    return int(bool(value))


def _optional_millionths(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return to_millionths(value)
    except ValueError:
        return None


def _event(market: Mapping[str, Any]) -> Mapping[str, Any]:
    events = first(market, "events", default=[]) or []
    return events[0] if events and isinstance(events[0], Mapping) else {}


def _market_category(market: Mapping[str, Any]) -> str | None:
    event = _event(market)
    direct = first(event, "category", default=first(market, "category"))
    if direct:
        return str(direct).lower()
    known = {
        "politics", "elections", "geopolitics", "sports", "crypto", "finance",
        "economy", "business", "tech", "science", "weather", "culture", "world",
    }
    tags = first(market, "tags", default=[]) or []
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        slug = str(first(tag, "slug", default="")).lower()
        label = str(first(tag, "label", default="")).lower()
        if slug in known:
            return slug
        if label in known:
            return label
    return None


def _market_times(market: Mapping[str, Any]) -> tuple[int | None, int | None]:
    return (
        parse_timestamp(first(market, "startDate", "start_date", "startDateIso")),
        parse_timestamp(first(market, "endDate", "end_date", "endDateIso")),
    )


def score_trade(
    trade: Mapping[str, Any],
    wallet: Mapping[str, Any],
    features: Mapping[str, Any],
) -> tuple[int, str, list[dict[str, Any]]]:
    components: list[dict[str, Any]] = []

    def add(rule: str, points: int, value: Any) -> None:
        if points:
            components.append({"rule": rule, "points": points, "value": value})

    age = wallet.get("wallet_age_seconds")
    if age is not None and age < 7 * 86400:
        add("wallet_under_7d", 3, age)
    elif age is not None and age < 30 * 86400:
        add("wallet_under_30d", 2, age)

    prior = wallet.get("prior_trade_count")
    window_complete = "history_window_complete" in (wallet.get("missing_reason") or "")
    if prior is not None and window_complete:
        if prior <= 5:
            add("recent_30d_trades_at_most_5", 2, prior)
        elif prior <= 20:
            add("recent_30d_trades_at_most_20", 1, prior)

    notional = int(trade["notional_microusd"])
    if notional >= 5_000_000_000:
        add("notional_at_least_5000", 2, notional)
    elif notional >= 1_000_000_000:
        add("notional_at_least_1000", 1, notional)

    ratio = features.get("size_vs_median_milli")
    if ratio is not None and ratio >= 10_000:
        add("size_at_least_10x_median", 2, ratio)
    elif ratio is not None and ratio >= 3_000:
        add("size_at_least_3x_median", 1, ratio)

    if int(trade["price_ppm"]) <= 250_000:
        add("price_at_most_025", 1, trade["price_ppm"])

    remaining = features.get("seconds_to_resolution")
    if remaining is not None and 0 <= remaining <= 7 * 86400:
        add("resolution_within_7d", 2, remaining)
    elif remaining is not None and 0 <= remaining <= 30 * 86400:
        add("resolution_within_30d", 1, remaining)

    prior_category = features.get("category_prior")
    if prior_category == "high":
        add("high_prior_category", 2, prior_category)
    elif prior_category == "medium":
        add("medium_prior_category", 1, prior_category)

    young = int(features.get("young_wallet_buys_1h") or 0)
    if young >= 3:
        add("young_wallet_cluster", 2, young)

    total = sum(component["points"] for component in components)
    bucket = "high" if total >= 8 else "medium" if total >= 5 else "low"
    return total, bucket, components


def parse_book(raw: Mapping[str, Any]) -> dict[str, Any]:
    def levels(side: str) -> list[tuple[int, int]]:
        parsed: list[tuple[int, int]] = []
        for row in raw.get(side, []) or []:
            if not isinstance(row, Mapping):
                continue
            try:
                parsed.append((to_millionths(row.get("price")), to_millionths(row.get("size"))))
            except ValueError:
                continue
        return sorted(parsed, key=lambda item: item[0], reverse=side == "bids")

    bids = levels("bids")
    asks = levels("asks")
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None

    def depth_within(delta_ppm: int) -> int | None:
        if best_ask is None:
            return None
        return sum(size for price, size in asks if price <= best_ask + delta_ppm)

    def average_fill(budget_microusd: int) -> int | None:
        remaining = budget_microusd
        cost = 0
        shares = 0
        for price, available_shares in asks:
            level_cost = price * available_shares // 1_000_000
            if level_cost <= remaining:
                take_shares = available_shares
                take_cost = level_cost
            else:
                take_shares = remaining * 1_000_000 // price if price else 0
                take_cost = price * take_shares // 1_000_000
            shares += take_shares
            cost += take_cost
            remaining -= take_cost
            if remaining <= 1:
                break
        if remaining > 1 or shares == 0:
            return None
        return cost * 1_000_000 // shares

    return {
        "bids": bids,
        "asks": asks,
        "best_bid_ppm": best_bid,
        "best_ask_ppm": best_ask,
        "spread_ppm": best_ask - best_bid if best_ask is not None and best_bid is not None else None,
        "ask_depth_1c_microshares": depth_within(10_000),
        "ask_depth_5c_microshares": depth_within(50_000),
        "avg_fill_100_ppm": average_fill(100_000_000),
        "avg_fill_500_ppm": average_fill(500_000_000),
        "avg_fill_1000_ppm": average_fill(1_000_000_000),
    }


class Collector:
    def __init__(self, db: Database, api: PolymarketAPI, *, clock=time.time, enrich_inline=False):
        self.db = db
        self.api = api
        self.clock = clock
        self.enrich_inline = enrich_inline

    def _now(self) -> int:
        return int(self.clock())

    def _fetch_market(self, condition_id: str) -> dict[str, Any]:
        market = self.api.market(condition_id)
        if market is None:
            raise APIError(f"no Gamma market for {condition_id}", retryable=True)
        return market

    def _cached_or_fetch_market(self, condition_id: str) -> dict[str, Any]:
        cached = self.db.row(
            "SELECT last_refreshed_at FROM markets WHERE condition_id=?", (condition_id,)
        )
        if cached and self._now() - int(cached["last_refreshed_at"]) < MARKET_CACHE_SECONDS:
            snapshot = self.db.row(
                """SELECT raw_json FROM market_snapshots
                   WHERE condition_id=? ORDER BY observed_at DESC LIMIT 1""",
                (condition_id,),
            )
            if snapshot:
                return json.loads(snapshot["raw_json"])
        return self._fetch_market(condition_id)

    def _persist_market_and_outcome(
        self, trade: Mapping[str, Any], market: Mapping[str, Any], now: int
    ) -> int:
        event = _event(market)
        start_ts, end_ts = _market_times(market)
        fee_schedule = first(market, "feeSchedule", "fee_schedule", default={}) or {}
        condition_id = str(first(market, "conditionId", "condition_id", default=trade["condition_id"])).lower()
        title = str(first(market, "question", "title", default=trade["title"]))
        slug = str(first(market, "slug", default=trade["slug"]))
        event_slug = str(first(event, "slug", default=trade["event_slug"])) or None
        is_sports = bool(first(market, "sportsMarketType", "sports_market_type", "gameStartTime"))
        market_values = (
            condition_id,
            str(first(event, "id", default="")) or None,
            title,
            slug,
            event_slug,
            _market_category(market),
            start_ts,
            end_ts,
            _bool(is_sports),
            _bool(first(market, "acceptingOrders", "accepting_orders", default=False)),
            _bool(first(market, "closed", default=False)),
            _bool(first(market, "feesEnabled", "fees_enabled", default=False)),
            _optional_millionths(first(fee_schedule, "rate")),
            now,
            now,
        )
        raw_json = canonical_json(market)
        digest = payload_hash(market)
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO markets(
                       condition_id,event_id,title,slug,event_slug,category,start_ts,end_ts,
                       is_sports,accepting_orders,closed,fees_enabled,fee_rate_ppm,
                       first_seen_at,last_refreshed_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(condition_id) DO UPDATE SET
                       event_id=excluded.event_id,title=excluded.title,slug=excluded.slug,
                       event_slug=excluded.event_slug,category=excluded.category,
                       start_ts=excluded.start_ts,end_ts=excluded.end_ts,
                       is_sports=excluded.is_sports,accepting_orders=excluded.accepting_orders,
                       closed=excluded.closed,fees_enabled=excluded.fees_enabled,
                       fee_rate_ppm=excluded.fee_rate_ppm,last_refreshed_at=excluded.last_refreshed_at""",
                market_values,
            )
            self.db.connection.execute(
                """INSERT INTO outcomes(asset_id,condition_id,outcome_name,outcome_index,first_seen_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(asset_id) DO UPDATE SET
                       outcome_name=excluded.outcome_name,
                       outcome_index=COALESCE(excluded.outcome_index,outcomes.outcome_index)""",
                (
                    trade["asset_id"], condition_id, trade["outcome_name"],
                    trade["outcome_index"] if trade["outcome_index"] != 999 else None, now,
                ),
            )
            self.db.connection.execute(
                """INSERT OR IGNORE INTO market_snapshots(
                       condition_id,observed_at,payload_hash,liquidity_microusd,
                       volume_24h_microusd,best_bid_ppm,best_ask_ppm,last_price_ppm,
                       one_hour_change_ppm,one_day_change_ppm,raw_json
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    condition_id,
                    now,
                    digest,
                    _optional_millionths(first(market, "liquidityNum", "liquidity_num", "liquidity")),
                    _optional_millionths(first(market, "volume24hr", "volume_24hr")),
                    _optional_millionths(first(market, "bestBid", "best_bid")),
                    _optional_millionths(first(market, "bestAsk", "best_ask")),
                    _optional_millionths(first(market, "lastTradePrice", "last_trade_price")),
                    _optional_millionths(first(market, "oneHourPriceChange", "one_hour_price_change")),
                    _optional_millionths(first(market, "oneDayPriceChange", "one_day_price_change")),
                    raw_json,
                ),
            )
        row = self.db.row(
            "SELECT snapshot_id FROM market_snapshots WHERE condition_id=? AND payload_hash=?",
            (condition_id, digest),
        )
        assert row is not None
        return int(row["snapshot_id"])

    @staticmethod
    def _selected_for_enrichment(trade: Mapping[str, Any]) -> bool:
        return int(trade["notional_microusd"]) >= ENRICH_NOTIONAL_MICROUSD or deterministic_sample(
            str(trade["trade_key"])
        )

    def process_trade(self, raw: Mapping[str, Any], source_mode: str, run_id: int, counts: Counts,
                      market_raw: Mapping[str, Any] | None = None) -> None:
        counts.seen += 1
        try:
            trade = normalize_trade(raw)
        except (ValueError, TypeError) as exc:
            counts.errors += 1
            self.db.record_error(run_id, "normalize_trade", str(exc))
            return
        if trade["side"] != "BUY" or trade["notional_microusd"] < MIN_NOTIONAL_MICROUSD:
            return
        if self.db.is_excluded(trade["condition_id"]):
            counts.excluded += 1
            return
        try:
            if market_raw is not None and first(market_raw, "conditionId", "condition_id") == trade["condition_id"]:
                market = market_raw
            else:
                market = self._cached_or_fetch_market(trade["condition_id"])
        except APIError as exc:
            counts.errors += 1
            self.db.upsert_pending("classify_market", trade["condition_id"], str(exc), self._now())
            self.db.record_error(
                run_id,
                "classify_market",
                str(exc),
                endpoint="gamma/markets",
                entity_type="condition_id",
                entity_id=trade["condition_id"],
                retryable=exc.retryable,
                status_code=exc.status,
            )
            return
        excluded, evidence = classify_five_minute_updown(trade, market)
        if excluded:
            self.db.exclude_market(trade["condition_id"], evidence or "unknown", self._now())
            self.db.resolve_pending("classify_market", trade["condition_id"])
            counts.excluded += 1
            return

        now = self._now()
        try:
            market_snapshot_id = self._persist_market_and_outcome(trade, market, now)
        except sqlite3.IntegrityError as exc:
            counts.errors += 1
            self.db.record_error(run_id, "persist_market", str(exc), entity_id=trade["condition_id"])
            return
        selected = self._selected_for_enrichment(trade)
        with self.db.connection:
            cursor = self.db.connection.execute(
                """INSERT OR IGNORE INTO trades(
                       trade_key,source_trade_id,fingerprint,transaction_hash,proxy_wallet,
                       condition_id,asset_id,side,price_ppm,size_microshares,notional_microusd,
                       trade_ts,ingested_at,source_mode,raw_json,enrichment_status
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade["trade_key"], trade["source_trade_id"], trade["fingerprint"],
                    trade["transaction_hash"], trade["proxy_wallet"], trade["condition_id"],
                    trade["asset_id"], trade["side"], trade["price_ppm"],
                    trade["size_microshares"], trade["notional_microusd"], trade["trade_ts"],
                    now, source_mode, canonical_json(raw), "pending" if selected else "not_selected",
                ),
            )
        if cursor.rowcount == 0:
            counts.duplicates += 1
            return
        counts.inserted += 1
        self.db.resolve_pending("classify_market", trade["condition_id"])
        if source_mode == "backfill":
            self._historical_book_placeholder(trade)
        if source_mode == "live" and trade["notional_microusd"] >= BOOK_NOTIONAL_MICROUSD:
            self.capture_book(trade["trade_key"], run_id)
        if selected and self.enrich_inline:
            if self.enrich_trade(trade["trade_key"], market, market_snapshot_id, run_id):
                counts.enriched += 1
                score = self.db.row(
                    "SELECT score,bucket FROM scores WHERE trade_key=? AND score_version=?",
                    (trade["trade_key"], SCORE_VERSION),
                )
                if score and score["bucket"] == "high":
                    counts.high_scores.append((trade["trade_key"], score["score"], trade["title"]))
                if source_mode == "live" and (
                    trade["notional_microusd"] >= BOOK_NOTIONAL_MICROUSD
                    or (score and score["score"] >= 5)
                ):
                    self.capture_book(trade["trade_key"], run_id)

    def _historical_book_placeholder(self, trade: Mapping[str, Any]) -> None:
        now = self._now()
        with self.db.connection:
            self.db.connection.execute(
                """INSERT OR IGNORE INTO order_book_snapshots(
                       trade_key,asset_id,snapshot_kind,requested_at,status,error_message
                   ) VALUES (?,?,'post_detection',?,'historical_unavailable',
                       'Historical order book was not observed')""",
                (trade["trade_key"], trade["asset_id"], now),
            )

    def _wallet_history_fetch(self, trade: Mapping[str, Any], start: int, end: int) -> sqlite3.Row:
        saved = self.db.row(
            """SELECT * FROM wallet_history_fetches WHERE proxy_wallet=? AND window_start=?
               AND window_end=? AND status IN ('complete','truncated') ORDER BY fetch_id DESC LIMIT 1""",
            (trade["proxy_wallet"], start, end),
        )
        if saved:
            print(f"  reusing stored wallet history fetch {saved['fetch_id']}", flush=True)
            return saved
        with self.db.connection:
            cursor = self.db.connection.execute(
                """INSERT INTO wallet_history_fetches(proxy_wallet,window_start,window_end,status,started_at)
                   VALUES (?,?,?,'fetching',?)""", (trade["proxy_wallet"], start, end, self._now()),
            )
        fetch_id = int(cursor.lastrowid)
        pending: list[tuple[int, int, str]] = []
        def flush_rows() -> None:
            if pending:
                with self.db.connection:
                    self.db.connection.executemany(
                        "INSERT INTO wallet_history_raw(fetch_id,row_number,raw_json) VALUES (?,?,?)", pending,
                    )
                pending.clear()
        status = "complete"
        error = None
        try:
            for index, raw in enumerate(self.api.iter_wallet_trades(trade["proxy_wallet"], end, start=start)):
                pending.append((fetch_id, index, canonical_json(raw)))
                if len(pending) >= 100:
                    flush_rows()
        except HistoryTruncated as exc:
            status, error = "truncated", str(exc)
            print("  history capped: raw sample retained; lifetime age and totals remain unknown", flush=True)
        except BaseException as exc:
            status, error = "failed", str(exc) or type(exc).__name__
            raise
        finally:
            flush_rows()
            with self.db.connection:
                self.db.connection.execute(
                    "UPDATE wallet_history_fetches SET status=?,finished_at=?,error_message=? WHERE fetch_id=?",
                    (status, self._now(), error, fetch_id),
                )
        return self.db.row("SELECT * FROM wallet_history_fetches WHERE fetch_id=?", (fetch_id,))

    def _wallet_snapshot(self, trade: Mapping[str, Any]) -> int:
        cutoff = int(trade["trade_ts"]) - 1
        window_start = max(1, int(trade["trade_ts"]) - 30 * 86400)
        # Preserve raw rows before calculating features, including for older snapshots.
        fetched = self._wallet_history_fetch(trade, window_start, cutoff)
        existing = self.db.row(
            """SELECT wallet_snapshot_id,missing_reason FROM wallet_snapshots
               WHERE proxy_wallet=? AND as_of_ts=? AND feature_version=?""",
            (trade["proxy_wallet"], trade["trade_ts"], FEATURE_VERSION),
        )
        if existing and f"raw_history_fetch_id:{fetched['fetch_id']}" in (existing["missing_reason"] or "").split(";"):
            return int(existing["wallet_snapshot_id"])
        notionals: list[int] = []
        markets: set[str] = set()
        malformed = 0
        truncated = fetched["status"] == "truncated"
        rows = self.db.connection.execute(
            "SELECT raw_json FROM wallet_history_raw WHERE fetch_id=? ORDER BY row_number", (fetched["fetch_id"],),
        )
        for raw in rows:
            try:
                prior = normalize_trade(json.loads(raw["raw_json"]))
            except (ValueError, TypeError):
                malformed += 1
                continue
            if not window_start <= prior["trade_ts"] <= cutoff:
                continue
            notionals.append(prior["notional_microusd"])
            markets.add(prior["condition_id"])
        now = self._now()
        completeness = "partial"
        reasons = ["resolved_performance_not_computed", "lifetime_age_and_counts_unknown",
                   f"history_window_start:{window_start}", f"history_window_end:{cutoff}",
                   "history_truncated_at_100_pages" if truncated else
                   "history_window_incomplete_malformed" if malformed else "history_window_complete",
                   f"raw_history_fetch_id:{fetched['fetch_id']}"]
        if malformed:
            reasons.append(f"malformed_prior_rows:{malformed}")
        with self.db.connection:
            cursor = self.db.connection.execute(
                """INSERT INTO wallet_snapshots(
                       proxy_wallet,as_of_ts,feature_version,first_activity_ts,wallet_age_seconds,
                       prior_trade_count,prior_market_count,mean_notional_microusd,
                       median_notional_microusd,max_notional_microusd,prior_resolved_count,
                       prior_wins,prior_pnl_microusd,completeness,missing_reason,computed_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(proxy_wallet,as_of_ts,feature_version) DO UPDATE SET
                       first_activity_ts=excluded.first_activity_ts,wallet_age_seconds=excluded.wallet_age_seconds,
                       prior_trade_count=excluded.prior_trade_count,prior_market_count=excluded.prior_market_count,
                       mean_notional_microusd=excluded.mean_notional_microusd,
                       median_notional_microusd=excluded.median_notional_microusd,
                       max_notional_microusd=excluded.max_notional_microusd,
                       completeness=excluded.completeness,missing_reason=excluded.missing_reason,
                       computed_at=excluded.computed_at""",
                (
                    trade["proxy_wallet"], trade["trade_ts"], FEATURE_VERSION, None, None,
                    len(notionals), len(markets), sum(notionals) // len(notionals) if notionals else None,
                    median_int(notionals), max(notionals) if notionals else None,
                    None, None, None, completeness, ";".join(reasons), now,
                ),
            )
        saved = self.db.row(
            "SELECT wallet_snapshot_id FROM wallet_snapshots WHERE proxy_wallet=? AND as_of_ts=? AND feature_version=?",
            (trade["proxy_wallet"], trade["trade_ts"], FEATURE_VERSION),
        )
        return int(saved["wallet_snapshot_id"])

    def enrich_trade(
        self,
        trade_key: str,
        market_raw: Mapping[str, Any] | None = None,
        market_snapshot_id: int | None = None,
        run_id: int | None = None,
    ) -> bool:
        row = self.db.row(
            """SELECT t.*,m.category,m.end_ts FROM trades t
               JOIN markets m ON m.condition_id=t.condition_id WHERE t.trade_key=?""",
            (trade_key,),
        )
        if not row:
            return False
        trade = dict(row)
        try:
            if market_raw is None:
                market_raw = self._cached_or_fetch_market(trade["condition_id"])
            if market_snapshot_id is None:
                snap = self.db.row(
                    """SELECT snapshot_id FROM market_snapshots
                       WHERE condition_id=? ORDER BY observed_at DESC LIMIT 1""",
                    (trade["condition_id"],),
                )
                market_snapshot_id = int(snap["snapshot_id"]) if snap else None
            wallet_snapshot_id = self._wallet_snapshot(trade)
            wallet = self.db.row(
                "SELECT * FROM wallet_snapshots WHERE wallet_snapshot_id=?", (wallet_snapshot_id,)
            )
            assert wallet is not None
            median = wallet["median_notional_microusd"]
            size_ratio = (
                trade["notional_microusd"] * 1000 // median if median and median > 0 else None
            )
            cluster = self.db.row(
                """SELECT COUNT(*) AS buys, COALESCE(SUM(t.notional_microusd),0) AS notional,
                          COALESCE(SUM(CASE WHEN ws.wallet_age_seconds < 604800 THEN 1 ELSE 0 END),0) AS young
                   FROM trades t
                   LEFT JOIN wallet_snapshots ws
                     ON ws.proxy_wallet=t.proxy_wallet AND ws.as_of_ts=t.trade_ts
                        AND ws.feature_version=?
                   WHERE t.asset_id=? AND t.trade_ts>=? AND t.trade_ts<?""",
                (FEATURE_VERSION, trade["asset_id"], trade["trade_ts"] - 3600, trade["trade_ts"]),
            )
            prior = category_prior(trade["category"], market_raw)
            missing = ["prior_resolved_count", "prior_wins", "prior_pnl_microusd",
                       "wallet_age_seconds", "first_activity_ts", "lifetime_trade_count",
                       "lifetime_market_count", "young_wallet_buys_1h"]
            if "history_window_complete" not in (wallet["missing_reason"] or ""):
                missing.append("complete_30d_history")
            if median is None:
                missing.append("size_vs_median_milli")
            if trade["end_ts"] is None:
                missing.append("seconds_to_resolution")
            features = {
                "size_vs_median_milli": size_ratio,
                "seconds_to_resolution": (
                    trade["end_ts"] - trade["trade_ts"] if trade["end_ts"] is not None else None
                ),
                "category_prior": prior,
                "young_wallet_buys_1h": int(cluster["young"]),
            }
            now = self._now()
            with self.db.connection:
                self.db.connection.execute(
                    """INSERT INTO trade_features(
                           trade_key,feature_version,wallet_snapshot_id,market_snapshot_id,
                           size_vs_median_milli,seconds_to_resolution,outcome_price_ppm,
                           same_asset_buys_1h,same_asset_notional_1h,young_wallet_buys_1h,
                           category_prior,computed_at,missing_fields_json
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(trade_key,feature_version) DO UPDATE SET
                           wallet_snapshot_id=excluded.wallet_snapshot_id,
                           market_snapshot_id=excluded.market_snapshot_id,
                           size_vs_median_milli=excluded.size_vs_median_milli,
                           seconds_to_resolution=excluded.seconds_to_resolution,
                           same_asset_buys_1h=excluded.same_asset_buys_1h,
                           same_asset_notional_1h=excluded.same_asset_notional_1h,
                           young_wallet_buys_1h=excluded.young_wallet_buys_1h,
                           category_prior=excluded.category_prior,
                           computed_at=excluded.computed_at,
                           missing_fields_json=excluded.missing_fields_json""",
                    (
                        trade_key, FEATURE_VERSION, wallet_snapshot_id, market_snapshot_id,
                        size_ratio, features["seconds_to_resolution"], trade["price_ppm"],
                        int(cluster["buys"]), int(cluster["notional"]), int(cluster["young"]),
                        prior, now, canonical_json(missing),
                    ),
                )
                score, bucket, components = score_trade(trade, dict(wallet), features)
                self.db.connection.execute(
                    """INSERT INTO scores(
                           trade_key,score_version,score,bucket,components_json,computed_at
                       ) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(trade_key,score_version) DO UPDATE SET
                           score=excluded.score,bucket=excluded.bucket,
                           components_json=excluded.components_json,computed_at=excluded.computed_at""",
                    (trade_key, SCORE_VERSION, score, bucket, canonical_json(components), now),
                )
                self.db.connection.execute(
                    "UPDATE trades SET enrichment_status='complete' WHERE trade_key=?", (trade_key,)
                )
            self.db.resolve_pending("enrich_trade", trade_key)
            return True
        except (APIError, ValueError, sqlite3.Error) as exc:
            with self.db.connection:
                self.db.connection.execute(
                    "UPDATE trades SET enrichment_status='failed' WHERE trade_key=?", (trade_key,)
                )
            self.db.upsert_pending("enrich_trade", trade_key, str(exc), self._now())
            self.db.record_error(
                run_id, "enrich_trade", str(exc), entity_type="trade", entity_id=trade_key,
                retryable=isinstance(exc, APIError) and exc.retryable,
                status_code=exc.status if isinstance(exc, APIError) else None,
            )
            return False

    def capture_book(self, trade_key: str, run_id: int | None = None) -> bool:
        trade = self.db.row("SELECT * FROM trades WHERE trade_key=?", (trade_key,))
        if not trade:
            return False
        requested = self._now()
        try:
            raw = self.api.book(trade["asset_id"])
            observed = self._now()
            parsed = parse_book(raw)
            with self.db.connection:
                cursor = self.db.connection.execute(
                    """INSERT INTO order_book_snapshots(
                           trade_key,asset_id,snapshot_kind,requested_at,observed_at,latency_ms,status,
                           best_bid_ppm,best_ask_ppm,spread_ppm,ask_depth_1c_microshares,
                           ask_depth_5c_microshares,avg_fill_100_ppm,avg_fill_500_ppm,
                           avg_fill_1000_ppm,raw_json
                       ) VALUES (?,?,'post_detection',?,?,?,'complete',?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(trade_key,snapshot_kind) DO NOTHING""",
                    (
                        trade_key, trade["asset_id"], requested, observed,
                        max(0, observed - trade["trade_ts"]) * 1000,
                        parsed["best_bid_ppm"], parsed["best_ask_ppm"], parsed["spread_ppm"],
                        parsed["ask_depth_1c_microshares"], parsed["ask_depth_5c_microshares"],
                        parsed["avg_fill_100_ppm"], parsed["avg_fill_500_ppm"],
                        parsed["avg_fill_1000_ppm"], canonical_json(raw),
                    ),
                )
                if cursor.rowcount:
                    snapshot_id = int(cursor.lastrowid)
                    for side in ("bids", "asks"):
                        side_name = "bid" if side == "bids" else "ask"
                        for level, (price, size) in enumerate(parsed[side]):
                            self.db.connection.execute(
                                """INSERT INTO order_book_levels(
                                       book_snapshot_id,side,level_number,price_ppm,size_microshares
                                   ) VALUES (?,?,?,?,?)""",
                                (snapshot_id, side_name, level, price, size),
                            )
            self.db.resolve_pending("capture_book", trade_key)
            return True
        except (APIError, ValueError, sqlite3.Error) as exc:
            with self.db.connection:
                self.db.connection.execute(
                    """INSERT INTO order_book_snapshots(
                           trade_key,asset_id,snapshot_kind,requested_at,status,error_message
                       ) VALUES (?,?,'post_detection',?,'failed',?)
                       ON CONFLICT(trade_key,snapshot_kind) DO UPDATE SET
                           status='failed',error_message=excluded.error_message""",
                    (trade_key, trade["asset_id"], requested, str(exc)),
                )
            self.db.upsert_pending("capture_book", trade_key, str(exc), self._now())
            self.db.record_error(
                run_id, "capture_book", str(exc), entity_type="trade", entity_id=trade_key,
                retryable=isinstance(exc, APIError) and exc.retryable,
                status_code=exc.status if isinstance(exc, APIError) else None,
            )
            return False

    def collect_window(
        self,
        *,
        start: int,
        end: int,
        source_mode: str,
        checkpoint_name: str,
        initial_cursor: str | None = None,
        condition: str | None = None,
        max_pages: int = 1000,
        market_raw: Mapping[str, Any] | None = None,
    ) -> Counts:
        counts = Counts()
        run_id = self.db.start_run(source_mode, self._now())
        cursor = initial_cursor
        status = "complete"
        latest_ts: int | None = None
        seen_cursors: set[str] = set()
        if cursor:
            seen_cursors.add(cursor)
        try:
            while True:
                print(f"Fetching {source_mode} page {counts.pages + 1} market={condition or 'global'}", flush=True)
                kwargs = {"start": start, "end": end, "cursor": cursor}
                if condition:
                    kwargs["condition"] = condition
                page: Page = self.api.trades_page(**kwargs)
                counts.pages += 1
                errors_before_page = counts.errors
                for raw in page.rows:
                    timestamp = first(raw, "timestamp")
                    if timestamp is not None:
                        if not start <= int(timestamp) <= end:
                            continue
                        latest_ts = max(latest_ts or int(timestamp), int(timestamp))
                    self.process_trade(raw, source_mode, run_id, counts, market_raw=market_raw)
                if counts.errors > errors_before_page:
                    # Retry the same page so failed rows cannot be skipped on resume.
                    self.db.save_checkpoint(checkpoint_name, cursor=cursor, window_start=start,
                                            window_end=end, last_trade_ts=latest_ts)
                    status = "failed"
                    print(f"  page {counts.pages}: inserted={counts.inserted} errors={counts.errors}; page retained for retry", flush=True)
                    break
                cursor = page.next_cursor
                self.db.save_checkpoint(
                    checkpoint_name,
                    cursor=cursor,
                    window_start=start,
                    window_end=end,
                    last_trade_ts=latest_ts,
                )
                print(f"  page {counts.pages}: inserted={counts.inserted} seen={counts.seen} errors={counts.errors}", flush=True)
                if not cursor:
                    break
                if cursor in seen_cursors:
                    raise APIError("Trade feed repeated a cursor; scan incomplete")
                seen_cursors.add(cursor)
                if counts.pages >= max_pages:
                    raise APIError("Trade scan reached page limit; rerun to resume", retryable=True)
        except (APIError, KeyboardInterrupt) as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if isinstance(exc, APIError):
                counts.errors += 1
                self.db.record_error(
                    run_id, "collect_page", str(exc), endpoint="data/v2/trades",
                    retryable=exc.retryable, status_code=exc.status,
                )
            else:
                raise
        finally:
            self.db.finish_run(run_id, status, counts.as_dict(), self._now())
        return counts

    def enrich_pending(self, limit: int | None = None) -> tuple[int, int]:
        sql = """SELECT t.trade_key FROM trades t
                 WHERE t.enrichment_status IN ('pending','failed')
                    OR (t.enrichment_status='complete' AND NOT EXISTS (
                        SELECT 1 FROM trade_features f WHERE f.trade_key=t.trade_key AND f.feature_version=?))
                    OR (t.enrichment_status='complete' AND NOT EXISTS (
                        SELECT 1 FROM wallet_history_fetches h WHERE h.proxy_wallet=t.proxy_wallet
                        AND h.window_start=MAX(1,t.trade_ts-2592000) AND h.window_end=t.trade_ts-1
                        AND h.status IN ('complete','truncated')))
                 ORDER BY t.trade_ts"""
        params: tuple[Any, ...] = (FEATURE_VERSION,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        complete = failed = 0
        for row in self.db.rows(sql, params):
            print(f"Enriching {row['trade_key']} ({complete + failed + 1})", flush=True)
            if self.enrich_trade(row["trade_key"]):
                complete += 1
            else:
                failed += 1
        return complete, failed

    def rescore(self) -> int:
        rows = self.db.rows(
            """SELECT t.*,tf.*,ws.wallet_age_seconds,ws.prior_trade_count,ws.missing_reason
               FROM trades t JOIN trade_features tf ON tf.trade_key=t.trade_key
               JOIN wallet_snapshots ws ON ws.wallet_snapshot_id=tf.wallet_snapshot_id
               WHERE tf.feature_version=?""",
            (FEATURE_VERSION,),
        )
        now = self._now()
        with self.db.connection:
            for row in rows:
                values = dict(row)
                score, bucket, components = score_trade(values, values, values)
                self.db.connection.execute(
                    """INSERT INTO scores(trade_key,score_version,score,bucket,components_json,computed_at)
                       VALUES (?,?,?,?,?,?) ON CONFLICT(trade_key,score_version) DO UPDATE SET
                       score=excluded.score,bucket=excluded.bucket,
                       components_json=excluded.components_json,computed_at=excluded.computed_at""",
                    (row["trade_key"], SCORE_VERSION, score, bucket, canonical_json(components), now),
                )
        return len(rows)
