from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from informed_flow.api import PolymarketAPI
from informed_flow.cli import main
from informed_flow.db import Database


class RecordingHTTP:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get_json(self, base, path, params=None):
        self.calls.append((base, path, params))
        return self.response


class APITests(unittest.TestCase):
    def test_v2_page_and_cursor(self) -> None:
        http = RecordingHTTP(
            {"data": [{"id": "one"}], "pagination": {"next_cursor": "cursor-2"}}
        )
        api = PolymarketAPI(http)
        page = api.trades_page(start=1, end=2)
        self.assertEqual(page.rows, [{"id": "one"}])
        self.assertEqual(page.next_cursor, "cursor-2")
        _, path, params = http.calls[0]
        self.assertEqual(path, "/v2/trades")
        self.assertEqual(params["side"], "BUY")
        self.assertEqual(params["filter_amount"], 100)

    def test_wallet_history_removes_live_feed_filters(self) -> None:
        http = RecordingHTTP({"data": [], "pagination": {"next_cursor": None}})
        api = PolymarketAPI(http)
        list(api.iter_wallet_trades("0xwallet", 99))
        params = http.calls[0][2]
        self.assertIsNone(params["side"])
        self.assertIsNone(params["filter_type"])
        self.assertEqual(params["end"], 99)


class CLITests(unittest.TestCase):
    def test_init_report_rescore_and_empty_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "data.sqlite3"
            output = Path(directory) / "exports"
            self.assertEqual(main(["--db", str(db_path), "init"]), 0)
            self.assertEqual(main(["--db", str(db_path), "report"]), 0)
            self.assertEqual(main(["--db", str(db_path), "rescore"]), 0)
            self.assertEqual(
                main(["--db", str(db_path), "export", "--output", str(output)]), 0
            )
            expected = {
                "model_features.csv",
                "trades.csv",
                "order_book_summaries.csv",
                "scores.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            with (output / "model_features.csv").open(encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertIn("trade_key", header)
            self.assertIn("wallet_age_seconds", header)

    def test_schema_contains_all_planned_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "data.sqlite3")
            db.initialize()
            names = {
                row["name"]
                for row in db.rows("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertTrue(
                {
                    "markets",
                    "market_exclusions",
                    "outcomes",
                    "market_snapshots",
                    "trades",
                    "wallet_snapshots",
                    "trade_features",
                    "scores",
                    "order_book_snapshots",
                    "order_book_levels",
                    "labels",
                    "collector_runs",
                    "checkpoints",
                    "pending_work",
                    "collection_errors",
                }.issubset(names)
            )
            db.close()


if __name__ == "__main__":
    unittest.main()
