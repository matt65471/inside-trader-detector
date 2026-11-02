from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from informed_flow.cli import main
from informed_flow.core import FEATURE_VERSION, SCORE_VERSION
from informed_flow.dataset import (
    FEATURE_COLUMNS,
    DatasetConfig,
    assign_bursts,
    assign_splits,
    build_dataset,
    sample_weight,
)
from informed_flow.db import Database

DAY = 86400
CUTOFF = 50 * DAY
COHORT = "test-cohort"


def _market_row(market: str, trade_ts: int, resolved_at: int | None) -> dict:
    return {"condition_id": market, "trade_ts": trade_ts, "resolved_at": resolved_at}


class SplitTests(unittest.TestCase):
    def test_before_after_and_overlap_rules(self) -> None:
        rows = [
            _market_row("train", 10 * DAY, 20 * DAY),
            _market_row("test", 60 * DAY, 70 * DAY),
            _market_row("overlap", 40 * DAY, 70 * DAY),
            _market_row("overlap", 55 * DAY, 70 * DAY),
            _market_row("unresolved", 40 * DAY, None),
        ]
        splits = assign_splits(rows, CUTOFF, seed=0)
        self.assertEqual(splits["train"][:2], ("train", "before_cutoff"))
        self.assertEqual(splits["test"], ("test", "after_cutoff", None))
        self.assertEqual(splits["overlap"][1], "overlap_random")
        self.assertEqual(splits["unresolved"][1], "overlap_random")

    def test_overlap_coin_flip_is_deterministic_and_roughly_even(self) -> None:
        rows = [_market_row(f"m{i}", 40 * DAY, 70 * DAY) for i in range(400)]
        first = assign_splits(rows, CUTOFF, seed=7)
        self.assertEqual(first, assign_splits(list(reversed(rows)), CUTOFF, seed=7))
        train = sum(value[0] == "train" for value in first.values())
        self.assertTrue(160 <= train <= 240, train)
        self.assertNotEqual(first, assign_splits(rows, CUTOFF, seed=8))

    def test_cv_folds_cover_only_train_markets_evenly(self) -> None:
        rows = [_market_row(f"m{i}", 10 * DAY, 20 * DAY) for i in range(23)]
        rows.append(_market_row("t", 60 * DAY, 70 * DAY))
        splits = assign_splits(rows, CUTOFF, seed=0)
        folds = [value[2] for key, value in splits.items() if key != "t"]
        self.assertEqual(sorted(set(folds)), [0, 1, 2, 3, 4])
        self.assertLessEqual(max(folds.count(f) for f in range(5)) - min(folds.count(f) for f in range(5)), 1)
        self.assertIsNone(splits["t"][2])

    def test_bursts_group_same_wallet_asset_within_sixty_seconds(self) -> None:
        rows = [
            {"trade_key": "a", "proxy_wallet": "w", "asset_id": "x", "trade_ts": 100},
            {"trade_key": "b", "proxy_wallet": "w", "asset_id": "x", "trade_ts": 150},
            {"trade_key": "c", "proxy_wallet": "w", "asset_id": "x", "trade_ts": 205},
            {"trade_key": "d", "proxy_wallet": "w", "asset_id": "x", "trade_ts": 400},
            {"trade_key": "e", "proxy_wallet": "w", "asset_id": "y", "trade_ts": 120},
        ]
        bursts = assign_bursts(rows)
        self.assertEqual(bursts["a"], ("a", 3))
        self.assertEqual(bursts["c"], ("a", 3))
        self.assertEqual(bursts["d"], ("d", 1))
        self.assertEqual(bursts["e"], ("e", 1))

    def test_sample_weight_reflects_enrichment_sampling(self) -> None:
        self.assertEqual(sample_weight(500_000_000), 1.0)
        self.assertEqual(sample_weight(499_999_999), 4.0)


class BuildDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "fixture.sqlite3"
        db = Database(self.db_path)
        db.initialize()
        self.conn = db.connection
        self.fetch = 0
        with self.conn:
            self.conn.execute(
                """INSERT INTO backfill_cohorts(cohort_name,window_start,window_end,days,category_limit,
                       seed,embedding_model,similarity_threshold_ppm,requested_device,phase,
                       created_at,updated_at)
                   VALUES (?,1,?,0,10,0,'m',900000,'cpu','complete',1,1)""",
                (COHORT, 200 * DAY),
            )
            self._market("train-m", "politics", resolved_at=20 * DAY)
            self._trade("t1", "train-m", 10 * DAY, 600, won=1)
            self._trade("t2", "train-m", 10 * DAY + 30, 100, won=1)
            self._trade("t3", "train-m", 11 * DAY, 100, won=1, enriched=False)
            self._market("test-m", "sports", resolved_at=80 * DAY)
            self._trade("t4", "test-m", 60 * DAY, 2000, won=0, truncated=True)
            self._market("overlap-m", "culture", resolved_at=70 * DAY)
            self._trade("t5", "overlap-m", 40 * DAY, 800, won=1)
            self._market("crypto-m", "crypto", resolved_at=20 * DAY)
            self._trade("t6", "crypto-m", 10 * DAY + 5, 900, won=1)
        db.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _market(self, market: str, category: str, *, resolved_at: int) -> None:
        self.conn.execute(
            """INSERT INTO markets(condition_id,title,slug,event_slug,start_ts,end_ts,
                   first_seen_at,last_refreshed_at,closed)
               VALUES (?,?,?,?,?,?,1,1,1)""",
            (market, market, market, f"event-{market}", 0, 100 * DAY),
        )
        self.conn.execute(
            "INSERT INTO outcomes(asset_id,condition_id,outcome_name,outcome_index,first_seen_at) VALUES (?,?,?,0,1)",
            (f"{market}-yes", market, "Yes"),
        )
        self.conn.execute(
            """INSERT INTO backfill_cohort_markets(cohort_name,condition_id,title,canonical_category,
                   raw_json,eligible,selection_status,fetch_status,discovered_at)
               VALUES (?,?,?,?,'{}',1,'selected','complete',1)""",
            (COHORT, market, market, category),
        )
        self.conn.execute(
            """INSERT INTO market_resolutions(condition_id,status,terminal,winning_outcome_index,
                   resolved_at,source,quality,raw_json,checked_at)
               VALUES (?,'resolved',1,0,?,'gamma_markets_v1','terminal_outcome_prices','{}',1)""",
            (market, resolved_at),
        )

    def _trade(self, key: str, market: str, ts: int, usd: int, *, won: int,
               enriched: bool = True, truncated: bool = False) -> None:
        notional = usd * 1_000_000
        self.conn.execute(
            """INSERT INTO trades(trade_key,fingerprint,transaction_hash,proxy_wallet,condition_id,
                   asset_id,side,price_ppm,size_microshares,notional_microusd,trade_ts,ingested_at,
                   source_mode,raw_json,enrichment_status)
               VALUES (?,?,?,?,?,?,'BUY',500000,?,?,?,1,'backfill','{}',?)""",
            (key, key, key, "wallet", market, f"{market}-yes", notional * 2, notional, ts,
             "complete" if enriched else "not_selected"),
        )
        for seconds in (900, 3600, 86400):
            self.conn.execute(
                """INSERT INTO labels(trade_key,horizon_seconds,scheduled_at,entry_price_ppm,outcome_won,
                       status,price_source,gross_pnl_microusd)
                   VALUES (?,?,?,600000,?,'complete','historical_price_history',?)""",
                (key, seconds, ts + seconds, won, (won * 1_000_000) - 600000),
            )
        if not enriched:
            return
        self.fetch += 1
        summary = {"observed_trade_count": 12, "distinct_market_count": 3, "active_days": 4,
                   "median_notional_microusd": 100_000_000, "mean_price_ppm": 400000,
                   "side_counts": {"BUY": 9, "SELL": 3},
                   "side_notional_microusd": {"BUY": 900, "SELL": 100}}
        self.conn.execute(
            """INSERT INTO wallet_history_fetches(fetch_id,proxy_wallet,window_start,window_end,status,started_at)
               VALUES (?,?,?,?,?,1)""",
            (self.fetch, "wallet", ts - 30 * DAY, ts - 1, "truncated" if truncated else "complete"),
        )
        self.conn.execute(
            "INSERT INTO wallet_history_summaries(fetch_id,summary_json) VALUES (?,?)",
            (self.fetch, json.dumps(summary)),
        )
        cursor = self.conn.execute(
            """INSERT INTO wallet_snapshots(proxy_wallet,as_of_ts,feature_version,prior_trade_count,
                   prior_market_count,completeness,missing_reason,computed_at)
               VALUES ('wallet',?,?,12,3,'partial',?,1)""",
            (ts, FEATURE_VERSION, f"history_window_complete;history_fetch_id:{self.fetch}"),
        )
        self.conn.execute(
            """INSERT INTO trade_features(trade_key,feature_version,wallet_snapshot_id,market_snapshot_id,
                   size_vs_median_milli,seconds_to_resolution,outcome_price_ppm,same_asset_buys_1h,
                   same_asset_notional_1h,young_wallet_buys_1h,category_prior,computed_at,missing_fields_json)
               VALUES (?,?,?,NULL,?,0,500000,2,5000000,0,'medium',1,'[]')""",
            (key, FEATURE_VERSION, cursor.lastrowid, usd * 10),
        )
        self.conn.execute(
            """INSERT INTO scores(trade_key,score_version,score,bucket,components_json,computed_at)
               VALUES (?,?,5,'medium','[]',1)""",
            (key, SCORE_VERSION),
        )

    def _build(self, name: str) -> tuple[dict, Path]:
        output = self.root / name
        manifest = build_dataset(DatasetConfig(
            db_path=self.db_path, cohort=COHORT, output=output, test_start=CUTOFF,
        ))
        return manifest, output

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_builds_split_files_without_leaking_labels(self) -> None:
        manifest, output = self._build("ds")
        features = {row["trade_key"]: row for row in self._rows(output / "features.csv")}
        self.assertEqual(sorted(features), ["t1", "t2", "t4", "t5"])
        self.assertEqual(manifest["dropped"], {"not_enriched": 1, "other_category": 1})
        self.assertEqual(tuple(self._rows(output / "features.csv")[0]), FEATURE_COLUMNS)
        for leaked in ("outcome_won", "edge", "resolved_at", "gross_pnl_per_share_24h",
                       "entry_price_24h", "best_bid_ppm", "last_price_ppm", "closed"):
            self.assertNotIn(leaked, FEATURE_COLUMNS)

        self.assertEqual(features["t1"]["split"], "train")
        self.assertEqual(features["t1"]["split_basis"], "before_cutoff")
        self.assertEqual(features["t4"]["split_basis"], "after_cutoff")
        self.assertEqual(features["t5"]["split_basis"], "overlap_random")
        self.assertEqual(features["t1"]["burst_id"], "t1")
        self.assertEqual(features["t2"]["burst_size"], "2")
        self.assertEqual(features["t1"]["market_weight"], "0.5")
        self.assertEqual(features["t1"]["sample_weight"], "1.0")
        self.assertEqual(features["t2"]["sample_weight"], "4.0")
        self.assertEqual(features["t4"]["history_truncated"], "1")
        self.assertEqual(features["t1"]["wallet_buy_notional_share"], "0.9")
        self.assertEqual(features["t1"]["hours_to_scheduled_end"], str(90 * 24.0))

        train = {row["trade_key"]: row for row in self._rows(output / "labels_train.csv")}
        test = {row["trade_key"]: row for row in self._rows(output / "labels_test.csv")}
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(features))
        self.assertIn("t4", test)
        self.assertEqual(test["t4"]["outcome_won"], "0")
        self.assertEqual(test["t4"]["edge"], "-0.5")
        self.assertEqual(train["t1"]["gross_pnl_per_share_24h"], "0.4")
        self.assertEqual(
            manifest["files"]["labels_test.csv"],
            hashlib.sha256((output / "labels_test.csv").read_bytes()).hexdigest(),
        )
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("Small cells", report)
        self.assertIn("| test | sports | 1 | 1 | sealed |", report)
        test_cells = [cell for cell in manifest["split_by_category"] if cell["split"] == "test"]
        self.assertTrue(test_cells)
        self.assertTrue(all(cell["win_rate"] is None for cell in test_cells))

    def test_rebuild_is_identical_and_database_is_untouched(self) -> None:
        before = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        first, _ = self._build("one")
        second, _ = self._build("two")
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(before, hashlib.sha256(self.db_path.read_bytes()).hexdigest())

    def test_refuses_to_overwrite_and_rejects_unknown_inputs(self) -> None:
        self._build("ds")
        with self.assertRaises(FileExistsError):
            self._build("ds")
        with self.assertRaises(ValueError):
            build_dataset(DatasetConfig(
                db_path=self.db_path, cohort="missing", output=self.root / "x", test_start=CUTOFF,
            ))

    def test_cli_builds_dataset_without_initializing_database(self) -> None:
        before = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        output = self.root / "cli"
        code = main(["--db", str(self.db_path), "build-dataset", "--cohort", COHORT,
                     "--output", str(output), "--test-start", "1970-02-20"])
        self.assertEqual(code, 0)
        self.assertTrue((output / "manifest.json").exists())
        self.assertEqual(before, hashlib.sha256(self.db_path.read_bytes()).hexdigest())
        with self.assertRaises(SystemExit):
            main(["--db", str(self.db_path), "build-dataset", "--cohort", COHORT,
                  "--output", str(self.root / "bad"), "--categories", "weather,astrology"])


if __name__ == "__main__":
    unittest.main()
