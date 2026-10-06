from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .api import PolymarketAPI
from .db import Database
from .service import Collector, Counts
from .api import APIError
from .core import classify_five_minute_updown, first, parse_timestamp
from .wallet_summary import compact_wallet_history
from .sampled_backfill import SampledBackfill, SampledBackfillConfig, TARGET_CATEGORIES

DEFAULT_DB = Path("data/informed_flow.sqlite3")


def _print_counts(label: str, counts: Counts) -> None:
    print(
        f"{label}: pages={counts.pages} seen={counts.seen} inserted={counts.inserted} "
        f"duplicates={counts.duplicates} excluded_updown_5m={counts.excluded} "
        f"enriched={counts.enriched} errors={counts.errors}"
    )
    for trade_key, score, title in counts.high_scores:
        print(f"  HIGH score={score} trade={trade_key} market={title}")


def _backfill(collector: Collector, db: Database, days: int, max_markets: int | None = None) -> int:
    # Separate namespace: old global-feed checkpoints cannot describe historical coverage.
    name = f"market_backfill_{days}d"
    checkpoint = db.checkpoint(name)
    end = int(checkpoint["window_end"]) if checkpoint else int(time.time())
    start = int(checkpoint["window_start"]) if checkpoint else end - days * 86400
    state = json.loads(checkpoint["cursor"]) if checkpoint and checkpoint["cursor"] else {
        "closed": True, "after": None, "skip": 0, "done": False,
    }
    def save_discovery() -> None:
        db.save_checkpoint(name, cursor=json.dumps(state), window_start=start,
                           window_end=end, last_trade_ts=None)
    save_discovery()
    print(f"Market backfill: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(start))} to "
          f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(end))}", flush=True)
    print("Coverage: Gamma-listed markets with Data API history; unavailable markets are not guaranteed. "
          "Enrichment is separate. Discovery scans all listed markets, including closed markets.", flush=True)
    examined = 0
    seen_discovery: set[tuple[bool, str | None]] = set()
    try:
        while not state["done"]:
            anchor = (state["closed"], state["after"])
            if anchor in seen_discovery:
                raise APIError("Gamma repeated a discovery cursor; scan incomplete")
            seen_discovery.add(anchor)
            print(f"Discovering {'closed' if state['closed'] else 'open'} markets; processed this run={examined}", flush=True)
            page = collector.api.markets_page(state["after"], closed=state["closed"])
            for market in page.rows[state["skip"]:]:
                if max_markets is not None and examined >= max_markets:
                    print("Test market limit reached; coverage is partial. Rerun to continue.", flush=True)
                    return 0
                condition = first(market, "conditionId", "condition_id")
                excluded, _ = classify_five_minute_updown(market, market)
                created = parse_timestamp(market.get("createdAt"))
                if condition and not excluded and (created is None or created <= end):
                    market_name = f"{name}:{condition}"
                    saved = db.checkpoint(market_name)
                    if not saved or saved["window_start"] != end:
                        counts = collector.collect_window(
                            start=start, end=end, source_mode="backfill", condition=str(condition),
                            checkpoint_name=market_name, initial_cursor=saved["cursor"] if saved else None,
                            market_raw=market,
                        )
                        _print_counts(f"market {condition}", counts)
                        saved = db.checkpoint(market_name)
                        if counts.errors or not saved or saved["cursor"]:
                            error = db.row("SELECT error_message FROM collection_errors ORDER BY error_id DESC LIMIT 1")
                            if error and counts.errors:
                                print(f"Market {condition}: {error['error_message']}", file=sys.stderr)
                            print("Market scan incomplete; rerun to retry/resume.", file=sys.stderr)
                            return 1
                        db.save_checkpoint(market_name, cursor=None, window_start=end,
                                           window_end=end, last_trade_ts=saved["last_trade_ts"])
                elif not condition:
                    raise APIError("Gamma listing contains a market without a condition ID; scan incomplete")
                examined += 1
                state["skip"] += 1
                save_discovery()
            if page.next_cursor:
                state["after"] = page.next_cursor
            elif state["closed"]:
                state["closed"] = False
                state["after"] = None
            else:
                state["done"] = True
            state["skip"] = 0
            save_discovery()
        print("Market scan complete for API-listed coverage. This frozen window is retained; use a new DB for a new window.", flush=True)
        return 0
    except APIError as exc:
        print(f"Backfill stopped: {exc}. Rerun to resume.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Backfill interrupted; saved trades and completed pages are retained.", flush=True)
        return 130


def _watch(collector: Collector, db: Database, *, once: bool, interval: int) -> int:
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    previous = signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            now = int(time.time())
            checkpoint = db.checkpoint("live")
            last_ts = int(checkpoint["last_trade_ts"]) if checkpoint and checkpoint["last_trade_ts"] else None
            start = max(1, (last_ts - 60) if last_ts else now - 120)
            counts = collector.collect_window(
                start=start,
                end=now,
                source_mode="live",
                checkpoint_name="live",
                initial_cursor=checkpoint["cursor"] if checkpoint else None,
            )
            _print_counts("live", counts)
            if once:
                return 1 if counts.errors and counts.pages == 0 else 0
            deadline = time.monotonic() + interval
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))
        print("Collector stopped cleanly.")
        return 0
    finally:
        signal.signal(signal.SIGINT, previous)


def _report(db: Database, high_limit: int, cohort: str | None = None) -> None:
    totals = db.row(
        """SELECT COUNT(*) AS trades,
                  COALESCE(SUM(source_mode='live'),0) AS live,
                  COALESCE(SUM(source_mode='backfill'),0) AS backfill,
                  COALESCE(SUM(enrichment_status='complete'),0) AS enriched,
                  COALESCE(SUM(enrichment_status='pending'),0) AS pending,
                  COALESCE(SUM(enrichment_status='failed'),0) AS failed,
                  MAX(ingested_at) AS freshest
           FROM trades"""
    )
    excluded = db.row("SELECT COUNT(*) AS n FROM market_exclusions")["n"]
    errors = db.row("SELECT COUNT(*) AS n FROM collection_errors")["n"]
    print("Informed-flow database")
    print(f"  trades: {totals['trades']} (live={totals['live']}, backfill={totals['backfill']})")
    print(
        f"  enrichment: complete={totals['enriched']} pending={totals['pending']} "
        f"failed={totals['failed']}"
    )
    print(f"  excluded five-minute Up/Down markets: {excluded}")
    print(f"  recorded errors: {errors}")
    history_rows = db.row("SELECT COUNT(*) n FROM wallet_history_raw")["n"]
    print(f"  raw wallet-history rows: {history_rows}")
    print(f"  detailed wallet summaries: {db.row('SELECT COUNT(*) n FROM wallet_history_summaries')['n']}")
    for row in db.rows("SELECT status,COUNT(*) n FROM wallet_history_fetches GROUP BY status ORDER BY status"):
        print(f"  wallet-history fetches {row['status']}: {row['n']}")
    if totals["freshest"]:
        print(f"  freshest ingestion: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(totals['freshest']))}")

    print("Score buckets")
    for row in db.rows(
        """SELECT bucket,COUNT(*) AS n,ROUND(AVG(score),2) AS mean_score
           FROM scores WHERE score_version=(SELECT MAX(score_version) FROM scores)
           GROUP BY bucket ORDER BY MIN(score)"""
    ):
        print(f"  {row['bucket']}: {row['n']} (mean={row['mean_score']})")

    print("Categories")
    for row in db.rows(
        """SELECT COALESCE(m.category,'unknown') AS category,COUNT(*) AS n
           FROM trades t JOIN markets m ON m.condition_id=t.condition_id
           GROUP BY COALESCE(m.category,'unknown') ORDER BY n DESC,category LIMIT 20"""
    ):
        print(f"  {row['category']}: {row['n']}")

    books = db.rows("SELECT status,COUNT(*) AS n FROM order_book_snapshots GROUP BY status ORDER BY status")
    if books:
        print("Book coverage")
        for row in books:
            print(f"  {row['status']}: {row['n']}")

    high = db.rows(
        """SELECT t.trade_ts,t.trade_key,m.title,o.outcome_name,t.notional_microusd,s.score
           FROM scores s JOIN trades t ON t.trade_key=s.trade_key
           JOIN markets m ON m.condition_id=t.condition_id
           JOIN outcomes o ON o.asset_id=t.asset_id
           WHERE s.bucket='high' AND s.score_version=(SELECT MAX(score_version) FROM scores)
           ORDER BY t.trade_ts DESC LIMIT ?""",
        (high_limit,),
    )
    if high:
        print("Recent high scores")
        for row in high:
            dollars = row["notional_microusd"] / 1_000_000
            print(
                f"  score={row['score']} ${dollars:,.2f} {row['outcome_name']} — "
                f"{row['title']} [{row['trade_key']}]"
            )

    if cohort:
        selected = db.row("SELECT * FROM backfill_cohorts WHERE cohort_name=?", (cohort,))
        if not selected:
            raise ValueError(f"Unknown cohort: {cohort}")
        print(f"Sampled cohort {cohort}")
        print(
            f"  phase={selected['phase']} window="
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(selected['window_start']))} to "
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(selected['window_end']))}"
        )
        print(
            f"  target/category={selected['category_limit']} seed={selected['seed']} "
            f"device={selected['effective_device'] or selected['requested_device']}"
        )
        if selected["last_error"]:
            print(f"  last note/error: {selected['last_error']}")
        if selected["embedding_device_log"] and selected["embedding_device_log"] != "[]":
            print(f"  embedding device fallbacks: {selected['embedding_device_log']}")
        for category in TARGET_CATEGORIES:
            row = db.row(
                """SELECT
                       COALESCE(SUM(eligible),0) eligible,
                       COALESCE(SUM(selection_status='selected'),0) selected,
                       COALESCE(SUM(selection_status='redundant'),0) redundant,
                       COALESCE(SUM(selection_status='selected' AND fetch_status='complete'),0) fetched,
                       COALESCE(SUM(selection_status='selected' AND fetch_status='failed'),0) failed,
                       COALESCE(SUM(selection_status='selected' AND fetch_status='complete'
                                    AND qualifying_trade_count=0),0) zero_trade,
                       COALESCE(SUM(CASE WHEN selection_status='selected'
                                        THEN qualifying_trade_count ELSE 0 END),0) trades,
                       COUNT(DISTINCT CASE WHEN selection_status='selected'
                             THEN COALESCE(event_id,event_slug) END) unique_events
                   FROM backfill_cohort_markets
                   WHERE cohort_name=? AND canonical_category=?""",
                (cohort, category),
            )
            wallets = db.row(
                """SELECT COUNT(DISTINCT t.proxy_wallet) n
                   FROM backfill_cohort_markets cm JOIN backfill_cohorts c USING(cohort_name)
                   JOIN trades t ON t.condition_id=cm.condition_id
                                  AND t.trade_ts BETWEEN c.window_start AND c.window_end
                   WHERE cm.cohort_name=? AND cm.canonical_category=?
                     AND cm.selection_status='selected'""",
                (cohort, category),
            )["n"]
            print(
                f"  {category}: eligible={row['eligible']} selected={row['selected']} "
                f"redundant={row['redundant']} fetched={row['fetched']} failed={row['failed']} "
                f"zero_trade={row['zero_trade']} trades={row['trades']} "
                f"events={row['unique_events']} wallets={wallets}"
            )


def _export_query(
    db: Database, output: Path, filename: str, query: str, params: tuple[Any, ...] = (),
) -> int:
    cursor = db.connection.execute(query, params)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    path = output / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")
    return len(rows)


def _export(db: Database, output: Path, cohort: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _export_query(db, output, "model_features.csv", "SELECT * FROM v_model_features ORDER BY trade_ts")
    _export_query(db, output, "trades.csv", "SELECT * FROM trades ORDER BY trade_ts")
    _export_query(
        db,
        output,
        "order_book_summaries.csv",
        """SELECT book_snapshot_id,trade_key,asset_id,snapshot_kind,requested_at,observed_at,
                  latency_ms,status,best_bid_ppm,best_ask_ppm,spread_ppm,
                  ask_depth_1c_microshares,ask_depth_5c_microshares,
                  avg_fill_100_ppm,avg_fill_500_ppm,avg_fill_1000_ppm,error_message
           FROM order_book_snapshots ORDER BY requested_at""",
    )
    _export_query(db, output, "scores.csv", "SELECT * FROM scores ORDER BY trade_key,score_version")
    _export_query(db, output, "wallet_summaries.csv",
                  "SELECT h.*,s.summary_json FROM wallet_history_fetches h JOIN wallet_history_summaries s USING(fetch_id) ORDER BY fetch_id")
    if cohort:
        if not db.row("SELECT 1 FROM backfill_cohorts WHERE cohort_name=?", (cohort,)):
            raise ValueError(f"Unknown cohort: {cohort}")
        _export_query(
            db, output, "cohort_markets.csv",
            """SELECT * FROM backfill_cohort_markets
               WHERE cohort_name=? AND selection_status='selected'
               ORDER BY canonical_category,random_rank""",
            (cohort,),
        )
        _export_query(
            db, output, "cohort_trades.csv",
            """SELECT t.*,cm.canonical_category,cm.random_rank
               FROM backfill_cohort_markets cm JOIN backfill_cohorts c USING(cohort_name)
               JOIN trades t ON t.condition_id=cm.condition_id
                              AND t.trade_ts BETWEEN c.window_start AND c.window_end
               WHERE cm.cohort_name=? AND cm.selection_status='selected'
               ORDER BY cm.canonical_category,cm.random_rank,t.trade_ts""",
            (cohort,),
        )
        _export_query(
            db, output, "cohort_similarity_rejections.csv",
            """SELECT cohort_name,condition_id,title,canonical_category,random_rank,
                      selection_reason,duplicate_of_condition_id,similarity_ppm
               FROM backfill_cohort_markets
               WHERE cohort_name=? AND selection_status='redundant'
               ORDER BY canonical_category,random_rank""",
            (cohort,),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="informed-flow", description="Read-only Polymarket informed-flow research collector"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Initialize or migrate the SQLite database")
    subparsers.add_parser("compact-wallet-history", help="Summarize stored enrichment history, delete its raw rows, and reclaim space; stop enrichment first")

    backfill = subparsers.add_parser("backfill", help="Backfill historical public trades")
    backfill.add_argument("--days", type=int, default=90)
    backfill.add_argument("--max-markets", type=int, help="Limit discovered markets for a partial test run")

    sampled = subparsers.add_parser(
        "sampled-backfill", help="Backfill a balanced, semantically deduplicated closed-market cohort"
    )
    sampled.add_argument("--cohort", required=True)
    sampled.add_argument("--days", type=int, default=90)
    sampled.add_argument("--markets-per-category", type=int, default=1000)
    sampled.add_argument("--seed", type=int, default=0)
    sampled.add_argument("--similarity-threshold", type=float, default=0.90)
    sampled.add_argument("--embedding-device", choices=("auto", "cuda", "mps", "cpu"), default="auto")

    watch = subparsers.add_parser("watch", help="Watch new public trades")
    watch.add_argument("--once", action="store_true", help="Poll once and exit")
    watch.add_argument("--interval", type=int, default=60, help="Seconds between polls")

    enrich = subparsers.add_parser("enrich", help="Retry selected incomplete enrichments")
    enrich.add_argument("--limit", type=int)

    subparsers.add_parser("rescore", help="Recompute versioned heuristic scores")
    report = subparsers.add_parser("report", help="Print dataset and score summaries")
    report.add_argument("--high-limit", type=int, default=10)
    report.add_argument("--cohort")
    export = subparsers.add_parser("export", help="Export portable CSV datasets")
    export.add_argument("--output", type=Path, default=Path("exports"))
    export.add_argument("--cohort")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "days", 1) <= 0:
        parser.error("--days must be positive")
    if getattr(args, "interval", 1) <= 0:
        parser.error("--interval must be positive")
    if getattr(args, "max_markets", None) is not None and args.max_markets <= 0:
        parser.error("--max-markets must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if getattr(args, "markets_per_category", 1) <= 0:
        parser.error("--markets-per-category must be positive")
    if not 0 < getattr(args, "similarity_threshold", 0.90) <= 1:
        parser.error("--similarity-threshold must be greater than 0 and at most 1")

    db = Database(args.db)
    try:
        db.initialize()
        if args.command == "init":
            print(f"Initialized {args.db}")
            return 0
        if args.command == "compact-wallet-history":
            summarized, deleted = compact_wallet_history(db)
            print(f"Created {summarized} summaries; removed {deleted} raw enrichment rows. Research trades retained.")
            return 0
        if args.command == "report":
            try:
                _report(db, args.high_limit, args.cohort)
                return 0
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1
        if args.command == "export":
            try:
                _export(db, args.output, args.cohort)
                return 0
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1

        collector = Collector(db, PolymarketAPI())
        if args.command == "backfill":
            return _backfill(collector, db, args.days, args.max_markets)
        if args.command == "sampled-backfill":
            return SampledBackfill(
                db,
                collector,
                SampledBackfillConfig(
                    cohort=args.cohort,
                    days=args.days,
                    markets_per_category=args.markets_per_category,
                    seed=args.seed,
                    similarity_threshold=args.similarity_threshold,
                    embedding_device=args.embedding_device,
                ),
            ).run()
        if args.command == "watch":
            return _watch(collector, db, once=args.once, interval=args.interval)
        if args.command == "enrich":
            complete, failed = collector.enrich_pending(args.limit)
            print(f"Enrichment retries: complete={complete} failed={failed}")
            return 1 if failed else 0
        if args.command == "rescore":
            count = collector.rescore()
            print(f"Rescored {count} trades")
            return 0
        parser.error(f"unknown command: {args.command}")
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
