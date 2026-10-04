from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from informed_flow.api import APIError, HistoryTruncated, Page, PolymarketAPI
from informed_flow.cli import _backfill, main
from informed_flow.db import Database
from informed_flow.service import Collector
from tests.test_service import FakeAPI, market, trade


class RecordingHTTP:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get_json(self, base, path, params=None):
        self.calls.append((base, path, params))
        return self.response


class APITests(unittest.TestCase):
    def test_wallet_window_and_page_cap_preserve_yielded_sample(self) -> None:
        class HistoryHTTP(RecordingHTTP):
            def get_json(self, base, path, params=None):
                self.calls.append((base, path, params))
                return {"data": [{"timestamp": 50}], "pagination": {"next_cursor": str(len(self.calls))}}
        http = HistoryHTTP(None)
        rows = []
        with patch("builtins.print"), self.assertRaises(HistoryTruncated):
            for row in PolymarketAPI(http).iter_wallet_trades("0xwallet", 99, start=10):
                rows.append(row)
        self.assertEqual(len(rows), 100)
        self.assertEqual(len(http.calls), 100)
        self.assertTrue(all(call[2]["start"] == 10 and call[2]["end"] == 99 for call in http.calls))

    def test_market_lookup_searches_closed_after_open_miss(self) -> None:
        class CohortHTTP(RecordingHTTP):
            def get_json(self, base, path, params=None):
                self.calls.append((base, path, params))
                return [market()] if params["closed"] == "true" else []
        http = CohortHTTP(None)
        self.assertEqual(PolymarketAPI(http).market("condition-1")["conditionId"], "condition-1")
        self.assertEqual([call[2]["closed"] for call in http.calls], ["false", "true"])

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

    def test_market_anchor_and_repeated_wallet_cursor(self) -> None:
        http = RecordingHTTP({"data": [], "pagination": {"next_cursor": "repeat"}})
        api = PolymarketAPI(http)
        api.trades_page(condition="0xcondition")
        self.assertEqual(http.calls[0][2]["condition"], "0xcondition")
        with self.assertRaisesRegex(APIError, "repeated"):
            list(api.iter_wallet_trades("0xwallet", 99))


class CLITests(unittest.TestCase):
    def test_market_backfill_resumes_partial_run_with_frozen_window(self) -> None:
        class HistoricalAPI(FakeAPI):
            def __init__(self):
                super().__init__()
                self.requests = []
            def market(self, condition_id):
                raise AssertionError("Backfill must reuse discovery metadata")
            def markets_page(self, cursor=None, *, closed=False):
                return Page([] if closed else [market(), market("condition-2")], None)
            def trades_page(self, **kwargs):
                self.requests.append(kwargs)
                condition = kwargs["condition"]
                return Page([trade(condition, condition=condition), trade("future", timestamp=3_000_000)], None)
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "test.sqlite3")
            db.initialize()
            api = HistoricalAPI()
            api.market_payloads["condition-2"] = market("condition-2")
            collector = Collector(db, api, clock=lambda: 2_000_100)
            with patch("informed_flow.cli.time.time", return_value=2_000_100):
                self.assertEqual(_backfill(collector, db, 1, max_markets=1), 0)
            with patch("informed_flow.cli.time.time", return_value=3_000_100):
                self.assertEqual(_backfill(collector, db, 1), 0)
            self.assertEqual([r["condition"] for r in api.requests], ["condition-1", "condition-2"])
            self.assertEqual([r["end"] for r in api.requests], [2_000_100, 2_000_100])
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 2)
            self.assertEqual(api.wallet_history_calls, 0)
            db.close()

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
