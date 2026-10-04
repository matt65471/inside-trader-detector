from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from informed_flow.api import Page
from informed_flow.db import Database
from informed_flow.service import Collector, Counts, parse_book, score_trade


def trade(
    trade_id: str = "t1",
    *,
    condition: str = "condition-1",
    asset: str = "asset-1",
    timestamp: int = 2_000_000,
    price: str = "0.25",
    size: str = "2400",
    slug: str = "company-announcement",
    title: str = "Will Company announce a product?",
) -> dict:
    return {
        "id": trade_id,
        "transaction_hash": f"tx-{trade_id}",
        "proxy_wallet": "0xwallet",
        "condition_id": condition,
        "asset_id": asset,
        "side": "BUY",
        "price": price,
        "size": size,
        "timestamp": timestamp,
        "title": title,
        "slug": slug,
        "event_slug": "company-event",
        "outcome": "Yes",
        "outcome_index": 0,
    }


def market(condition: str = "condition-1", slug: str = "company-announcement") -> dict:
    return {
        "id": "market-id",
        "conditionId": condition,
        "question": "Will Company announce a product?",
        "slug": slug,
        "startDate": "1970-01-20T00:00:00Z",
        "endDate": "1970-01-30T00:00:00Z",
        "acceptingOrders": True,
        "closed": False,
        "feesEnabled": True,
        "feeSchedule": {"rate": "0.04"},
        "liquidityNum": "10000",
        "volume24hr": "5000",
        "bestBid": "0.24",
        "bestAsk": "0.26",
        "lastTradePrice": "0.25",
        "events": [{"id": "event-1", "slug": "company-event", "category": "Business"}],
        "tags": [{"slug": "tech", "label": "Tech"}],
    }


class FakeAPI:
    def __init__(self) -> None:
        self.market_payloads = {"condition-1": market()}
        self.wallet_rows: list[dict] = []
        self.pages: list[Page] = []
        self.book_payload = {
            "bids": [{"price": "0.24", "size": "1000"}],
            "asks": [
                {"price": "0.26", "size": "1000"},
                {"price": "0.27", "size": "5000"},
            ],
        }

    def market(self, condition_id: str):
        return self.market_payloads.get(condition_id)

    def iter_wallet_trades(self, user: str, end: int):
        yield from self.wallet_rows

    def trades_page(self, **kwargs):
        return self.pages.pop(0)

    def book(self, asset_id: str):
        return self.book_payload


class BookAndScoreTests(unittest.TestCase):
    def test_book_levels_are_sorted_and_fill_requires_full_depth(self) -> None:
        parsed = parse_book(
            {
                "bids": [{"price": "0.20", "size": "1"}, {"price": "0.25", "size": "1"}],
                "asks": [{"price": "0.30", "size": "1000"}, {"price": "0.28", "size": "1000"}],
            }
        )
        self.assertEqual(parsed["best_bid_ppm"], 250_000)
        self.assertEqual(parsed["best_ask_ppm"], 280_000)
        self.assertEqual(parsed["spread_ppm"], 30_000)
        self.assertIsNotNone(parsed["avg_fill_100_ppm"])
        self.assertIsNone(parsed["avg_fill_1000_ppm"])

    def test_score_components_and_bucket(self) -> None:
        value, bucket, components = score_trade(
            {"notional_microusd": 5_000_000_000, "price_ppm": 200_000},
            {"wallet_age_seconds": 100, "prior_trade_count": 1},
            {
                "size_vs_median_milli": 12_000,
                "seconds_to_resolution": 3600,
                "category_prior": "high",
                "young_wallet_buys_1h": 3,
            },
        )
        self.assertEqual(bucket, "high")
        self.assertGreaterEqual(value, 8)
        self.assertEqual(value, sum(item["points"] for item in components))


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tempdir.name) / "collector.sqlite3")
        self.db.initialize()
        self.api = FakeAPI()
        self.collector = Collector(self.db, self.api, clock=lambda: 2_000_100)

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def test_process_is_idempotent_and_enriches_as_of_timestamp(self) -> None:
        prior = trade("prior", timestamp=1_000_000, price="0.50", size="200")
        future = trade("future", timestamp=3_000_000, price="0.50", size="99999")
        self.api.wallet_rows = [prior, future]
        counts = Counts()
        run_id = self.db.start_run("live", 2_000_100)
        raw = trade()
        self.collector.process_trade(raw, "live", run_id, counts)
        self.collector.process_trade(raw, "live", run_id, counts)

        self.assertEqual(counts.inserted, 1)
        self.assertEqual(counts.duplicates, 1)
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM markets")["n"], 1)
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
        wallet = self.db.row("SELECT * FROM wallet_snapshots")
        self.assertEqual(wallet["prior_trade_count"], 1)
        self.assertEqual(wallet["max_notional_microusd"], 100_000_000)
        self.assertEqual(
            self.db.row("SELECT enrichment_status FROM trades")["enrichment_status"], "complete"
        )

    def test_updown_is_blacklisted_before_research_rows(self) -> None:
        raw = trade(
            "updown",
            condition="updown-condition",
            asset="updown-asset",
            slug="btc-updown-5m-2000000",
            title="Bitcoin Up or Down",
        )
        self.api.market_payloads["updown-condition"] = market(
            "updown-condition", "btc-updown-5m-2000000"
        )
        counts = Counts()
        run_id = self.db.start_run("live", 2_000_100)
        self.collector.process_trade(raw, "live", run_id, counts)

        self.assertEqual(counts.excluded, 1)
        self.assertTrue(self.db.is_excluded("updown-condition"))
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM markets")["n"], 0)
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM trades")["n"], 0)

    def test_backfill_creates_historical_book_placeholder(self) -> None:
        self.api.wallet_rows = []
        counts = Counts()
        run_id = self.db.start_run("backfill", 2_000_100)
        self.collector.process_trade(trade(), "backfill", run_id, counts)
        book = self.db.row("SELECT status FROM order_book_snapshots")
        self.assertEqual(book["status"], "historical_unavailable")

    def test_collect_window_follows_cursor_and_saves_checkpoint(self) -> None:
        self.api.wallet_rows = []
        self.api.pages = [Page([trade()], "next"), Page([trade()], None)]
        counts = self.collector.collect_window(
            start=1_900_000,
            end=2_000_001,
            source_mode="live",
            checkpoint_name="live",
        )
        self.assertEqual(counts.pages, 2)
        self.assertEqual(counts.inserted, 1)
        self.assertEqual(counts.duplicates, 1)
        checkpoint = self.db.checkpoint("live")
        self.assertIsNone(checkpoint["cursor"])
        self.assertEqual(checkpoint["last_trade_ts"], 2_000_000)


if __name__ == "__main__":
    unittest.main()
