from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from informed_flow.core import (
    classify_five_minute_updown,
    deterministic_sample,
    normalize_trade,
    notional_microusd,
    to_millionths,
)
from informed_flow.db import Database


TRADE = {
    "id": "trade-1",
    "transaction_hash": "0xabc",
    "proxy_wallet": "0xABCDEF",
    "condition_id": "0xcondition",
    "asset_id": "asset-1",
    "side": "BUY",
    "price": "0.25",
    "size": "400",
    "timestamp": 1_700_000_000,
    "title": "Will something happen?",
    "slug": "will-something-happen",
    "outcome": "Yes",
    "outcome_index": 0,
}


class CoreTests(unittest.TestCase):
    def test_exact_numeric_conversion(self) -> None:
        self.assertEqual(to_millionths("0.123456"), 123456)
        self.assertEqual(notional_microusd("0.25", "400"), 100_000_000)

    def test_trade_normalization_and_fingerprint_are_stable(self) -> None:
        first = normalize_trade(TRADE)
        second = normalize_trade(dict(reversed(list(TRADE.items()))))
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["trade_key"], "trade-1")
        self.assertEqual(first["notional_microusd"], 100_000_000)

    def test_live_v2_token_id_is_accepted(self) -> None:
        live_shape = dict(TRADE)
        live_shape["token_id"] = live_shape.pop("asset_id")
        self.assertEqual(normalize_trade(live_shape)["asset_id"], "asset-1")

    def test_fingerprint_distinguishes_fills_in_same_transaction(self) -> None:
        first = normalize_trade(TRADE)
        changed = dict(TRADE, asset_id="asset-2")
        self.assertNotEqual(first["fingerprint"], normalize_trade(changed)["fingerprint"])

    def test_slug_excludes_five_minute_updown(self) -> None:
        trade = {"slug": "btc-updown-5m-1700000000", "title": "Bitcoin Up or Down"}
        self.assertEqual(classify_five_minute_updown(trade, {}), (True, "slug"))

    def test_title_and_duration_exclude_when_slug_is_missing(self) -> None:
        trade = {"title": "Ethereum Up or Down"}
        market = {"startDate": "2026-01-01T00:00:00Z", "endDate": "2026-01-01T00:05:00Z"}
        self.assertEqual(
            classify_five_minute_updown(trade, market), (True, "title_and_duration")
        )

    def test_longer_crypto_market_is_not_excluded(self) -> None:
        trade = {"title": "Bitcoin Up or Down"}
        market = {"startDate": "2026-01-01T00:00:00Z", "endDate": "2026-01-02T00:00:00Z"}
        self.assertEqual(classify_five_minute_updown(trade, market), (False, None))

    def test_sampling_is_deterministic(self) -> None:
        self.assertEqual(deterministic_sample("abc"), deterministic_sample("abc"))


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tempdir.name) / "test.sqlite3")
        self.db.initialize()

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def test_initialize_is_idempotent_and_enables_foreign_keys(self) -> None:
        self.db.initialize()
        version = self.db.row("SELECT COUNT(*) AS n FROM schema_migrations")
        self.assertEqual(version["n"], 1)
        self.assertEqual(self.db.row("PRAGMA foreign_keys")[0], 1)

    def test_exclusion_trigger_blocks_market(self) -> None:
        self.db.exclude_market("excluded", "slug", 100)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.db.connection:
                self.db.connection.execute(
                    """INSERT INTO markets(
                           condition_id,title,slug,is_sports,accepting_orders,closed,
                           fees_enabled,first_seen_at,last_refreshed_at
                       ) VALUES ('excluded','x','x',0,1,0,0,100,100)"""
                )

    def test_market_and_snapshot_uniqueness(self) -> None:
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO markets(
                       condition_id,title,slug,is_sports,accepting_orders,closed,
                       fees_enabled,first_seen_at,last_refreshed_at
                   ) VALUES ('market','x','x',0,1,0,0,100,100)"""
            )
            self.db.connection.execute(
                """INSERT INTO market_snapshots(
                       condition_id,observed_at,payload_hash,raw_json
                   ) VALUES ('market',100,'hash','{}')"""
            )
            self.db.connection.execute(
                """INSERT OR IGNORE INTO market_snapshots(
                       condition_id,observed_at,payload_hash,raw_json
                   ) VALUES ('market',101,'hash','{}')"""
            )
        count = self.db.row("SELECT COUNT(*) AS n FROM market_snapshots")
        self.assertEqual(count["n"], 1)


if __name__ == "__main__":
    unittest.main()
