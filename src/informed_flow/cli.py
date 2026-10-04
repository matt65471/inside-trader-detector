from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .api import PolymarketAPI
from .db import Database
from .service import Collector, Counts

DEFAULT_DB = Path("data/informed_flow.sqlite3")


def _print_counts(label: str, counts: Counts) -> None:
    print(
        f"{label}: pages={counts.pages} seen={counts.seen} inserted={counts.inserted} "
        f"duplicates={counts.duplicates} excluded_updown_5m={counts.excluded} "
        f"enriched={counts.enriched} errors={counts.errors}"
    )
    for trade_key, score, title in counts.high_scores:
        print(f"  HIGH score={score} trade={trade_key} market={title}")


def _backfill(collector: Collector, db: Database, days: int) -> int:
    requested_end = int(time.time())
    requested_start = requested_end - days * 86400
    checkpoint_name = f"backfill_{days}d"
    checkpoint = db.checkpoint(checkpoint_name)
    if checkpoint and checkpoint["window_start"] is not None:
        window_start = max(requested_start, int(checkpoint["window_start"]))
        window_end = int(checkpoint["window_end"] or min(window_start + 86400, requested_end))
        cursor = checkpoint["cursor"]
        if cursor is None and window_end < requested_end:
            window_start = window_end
            window_end = min(window_start + 86400, requested_end)
    else:
        window_start = requested_start
        window_end = min(window_start + 86400, requested_end)
        cursor = None

    while window_start < requested_end:
        label = f"backfill {time.strftime('%Y-%m-%d', time.gmtime(window_start))}"
        counts = collector.collect_window(
            start=window_start,
            end=window_end,
            source_mode="backfill",
            checkpoint_name=checkpoint_name,
            initial_cursor=cursor,
        )
        _print_counts(label, counts)
        checkpoint = db.checkpoint(checkpoint_name)
        if counts.errors and counts.pages == 0:
            print("Backfill stopped after an API failure; rerun to resume.", file=sys.stderr)
            return 1
        if checkpoint and checkpoint["cursor"]:
            print("Backfill page remains pending; rerun to resume.", file=sys.stderr)
            return 1
        window_start = window_end
        window_end = min(window_start + 86400, requested_end)
        cursor = None
        db.save_checkpoint(
            checkpoint_name,
            cursor=None,
            window_start=window_start,
            window_end=window_end,
            last_trade_ts=checkpoint["last_trade_ts"] if checkpoint else None,
        )
    print(f"Backfill complete through {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(requested_end))}")
    return 0


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


def _report(db: Database, high_limit: int) -> None:
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


def _export_query(db: Database, output: Path, filename: str, query: str) -> int:
    cursor = db.connection.execute(query)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    path = output / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")
    return len(rows)


def _export(db: Database, output: Path) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="informed-flow", description="Read-only Polymarket informed-flow research collector"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Initialize or migrate the SQLite database")

    backfill = subparsers.add_parser("backfill", help="Backfill historical public trades")
    backfill.add_argument("--days", type=int, default=90)

    watch = subparsers.add_parser("watch", help="Watch new public trades")
    watch.add_argument("--once", action="store_true", help="Poll once and exit")
    watch.add_argument("--interval", type=int, default=60, help="Seconds between polls")

    enrich = subparsers.add_parser("enrich", help="Retry selected incomplete enrichments")
    enrich.add_argument("--limit", type=int)

    subparsers.add_parser("rescore", help="Recompute versioned heuristic scores")
    report = subparsers.add_parser("report", help="Print dataset and score summaries")
    report.add_argument("--high-limit", type=int, default=10)
    export = subparsers.add_parser("export", help="Export portable CSV datasets")
    export.add_argument("--output", type=Path, default=Path("exports"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "days", 1) <= 0:
        parser.error("--days must be positive")
    if getattr(args, "interval", 1) <= 0:
        parser.error("--interval must be positive")

    db = Database(args.db)
    try:
        db.initialize()
        if args.command == "init":
            print(f"Initialized {args.db}")
            return 0
        if args.command == "report":
            _report(db, args.high_limit)
            return 0
        if args.command == "export":
            _export(db, args.output)
            return 0

        collector = Collector(db, PolymarketAPI())
        if args.command == "backfill":
            return _backfill(collector, db, args.days)
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

