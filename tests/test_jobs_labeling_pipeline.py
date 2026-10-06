from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from informed_flow.api import Page
from informed_flow.db import Database
from informed_flow.jobs import JobQueue
from informed_flow.labeling import HistoricalLabeler, ResolutionStore, resolution_result
from informed_flow.pipeline import AutomatedPipeline
from informed_flow.sampled_backfill import SampledBackfill, SampledBackfillConfig, normalize_title
from informed_flow.service import Collector, Counts
from tests.test_sampled_backfill import FakeEmbedder, SampleAPI, sampled_market
from tests.test_service import FakeAPI, market, trade


class LabelAPI(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.resolution_rows: list[dict] = []
        self.history_rows: list[dict] = []
        self.price_history_calls: list[tuple[str, int, int, int]] = []

    def resolutions(self, condition_ids: list[str]) -> list[dict]:
        return [row for row in self.resolution_rows if row["condition_id"] in condition_ids]

    def price_history(self, asset_id: str, start: int, end: int, *, fidelity: int = 1):
        self.price_history_calls.append((asset_id, start, end, fidelity))
        return [row for row in self.history_rows if start <= row["t"] <= end]


class JobQueueTests(unittest.TestCase):
    def test_unique_jobs_lease_recovery_dead_letter_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = [100]
            db = Database(Path(directory) / "queue.sqlite3")
            db.initialize()
            queue = JobQueue(db, clock=lambda: now[0])
            self.assertTrue(queue.enqueue("scan_market", "scan:a", {"condition": "a"}, max_attempts=2))
            self.assertFalse(queue.enqueue("scan_market", "scan:a", {"condition": "a"}))
            first = queue.claim("worker-a", ("scan_market",), lease_seconds=5)
            self.assertIsNotNone(first)
            now[0] = 106
            self.assertEqual(queue.recover_expired(), 1)
            second = queue.claim("worker-b", ("scan_market",), lease_seconds=5)
            self.assertEqual(second.attempts, 2)
            self.assertEqual(queue.fail(second, "worker-b", RuntimeError("boom")), "dead")
            self.assertEqual(queue.counts(), {"dead": 1})
            self.assertEqual(queue.retry_dead(kind="scan_market", entity_id="a"), 0)
            self.assertEqual(queue.retry_dead(kind="scan_market"), 1)
            retried = queue.claim("worker-c", ("scan_market",))
            queue.succeed(retried, "worker-c")
            self.assertEqual(queue.counts(), {"succeeded": 1})
            db.close()


class ResolutionAndLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tempdir.name) / "labels.sqlite3")
        self.db.initialize()
        self.api = LabelAPI()
        self.collector = Collector(self.db, self.api, clock=lambda: 2_200_000)

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def _store(self, raw: dict) -> None:
        self.collector.process_trade(
            raw, "backfill", self.db.start_run("backfill", 2_200_000), Counts(),
            market_raw=market(raw["condition_id"]),
        )

    def _resolve(self, *, resolved_at: int = 2_100_000) -> None:
        self.api.resolution_rows = [{
            "condition_id": "condition-1", "status": "resolved",
            "payouts": [1_000_000, 0], "resolved_at": resolved_at,
            "was_disputed": True, "reporter": "UMA_OO",
            "resolution_source": "reported",
        }]
        ResolutionStore(self.db, self.api, clock=lambda: 2_200_000).verify(
            "unused", ["condition-1"],
        )

    def test_resolution_accepts_micro_payouts_and_rejects_ambiguous_or_pending(self) -> None:
        self.assertEqual(
            resolution_result({"status": "resolved", "payouts": [1_000_000, 0]}),
            (True, 0, "resolved", "terminal_payout"),
        )
        self.assertEqual(
            resolution_result({"status": "resolved", "payouts": [500_000, 500_000]})[3],
            "ambiguous_payouts",
        )
        self.assertEqual(
            resolution_result({"status": "proposed", "payouts": [1_000_000, 0]})[3],
            "nonterminal_resolution",
        )
        self.assertEqual(
            resolution_result({
                "status": "canceled", "payouts": [1_000_000, 0],
                "resolved_at": "2026-01-01T00:00:00Z",
            })[3],
            "nonterminal_resolution",
        )

    def test_nonterminal_market_is_ineligible_before_embedding(self) -> None:
        rows = [
            sampled_market("terminal", "Terminal finance question", "Business", "event-a"),
            sampled_market("pending", "Pending finance question", "Business", "event-b"),
        ]
        discovery_api = SampleAPI(rows)
        config = SampledBackfillConfig("resolved-only", days=1, resolution_required=True)
        sampled = SampledBackfill(
            self.db, Collector(self.db, discovery_api, clock=lambda: 2_000_100), config,
            clock=lambda: 2_000_100, embedder=FakeEmbedder({}),
        )
        sampled.discover(sampled._cohort())
        self.api.resolution_rows = [
            {"condition_id": "terminal", "status": "resolved", "payouts": [1_000_000, 0]},
            {"condition_id": "pending", "status": "proposed", "payouts": [1_000_000, 0]},
        ]
        ResolutionStore(self.db, self.api).verify("resolved-only", ["terminal", "pending"])
        states = {
            row["condition_id"]: (row["eligible"], row["selection_status"], row["eligibility_reason"])
            for row in self.db.rows("SELECT * FROM backfill_cohort_markets")
        }
        self.assertEqual(states["terminal"][:2], (1, "candidate"))
        self.assertEqual(states["pending"], (0, "ineligible", "nonterminal_resolution"))

    def test_earliest_post_target_price_gross_pnl_unknown_fees_and_cache_reuse(self) -> None:
        self._store(trade("first"))
        self._store(trade("second", timestamp=2_000_100))
        self._resolve()
        self.api.history_rows = [
            {"t": 2_000_899, "p": 0.10},
            {"t": 2_000_960, "p": 0.40},
            {"t": 2_001_030, "p": 0.45},
            {"t": 2_001_060, "p": 0.50},
        ]
        labeler = HistoricalLabeler(self.db, self.api, clock=lambda: 2_200_000)
        labeler.observe("cohort", "first", 900)
        labeler.observe("cohort", "second", 900)
        labels = self.db.rows("SELECT * FROM labels ORDER BY trade_key")
        self.assertEqual([row["observed_at"] for row in labels], [2_000_960, 2_001_030])
        self.assertEqual([row["gross_pnl_microusd"] for row in labels], [600_000, 550_000])
        self.assertTrue(all(row["price_quality"] == "approximate" for row in labels))
        self.assertTrue(all(row["fee_source"] is None and row["net_pnl_microusd"] is None for row in labels))
        self.assertEqual(len(self.api.price_history_calls), 1)
        resolution = self.db.row("SELECT * FROM market_resolutions")
        self.assertEqual((resolution["terminal"], resolution["was_disputed"]), (1, 1))

    def test_never_uses_pre_target_or_more_than_five_minutes_late(self) -> None:
        self._store(trade())
        self._resolve()
        self.api.history_rows = [
            {"t": 2_000_899, "p": 0.10},
            {"t": 2_001_201, "p": 0.90},
        ]
        HistoricalLabeler(self.db, self.api).observe("cohort", "t1", 900)
        label = self.db.row("SELECT * FROM labels")
        self.assertEqual(label["status"], "price_unavailable")
        self.assertIsNone(label["entry_price_ppm"])

    def test_resolution_before_horizon_skips_price_lookup(self) -> None:
        self._store(trade())
        self._resolve(resolved_at=2_000_899)
        HistoricalLabeler(self.db, self.api).observe("cohort", "t1", 900)
        self.assertEqual(
            self.db.row("SELECT status FROM labels")["status"],
            "unavailable_before_horizon",
        )
        self.assertEqual(self.api.price_history_calls, [])

    def test_public_trade_print_is_the_approximate_fallback(self) -> None:
        self._store(trade())
        self._store(trade("later", timestamp=2_000_930, price="0.30"))
        self._resolve()
        HistoricalLabeler(self.db, self.api).observe("cohort", "t1", 900)
        label = self.db.row("SELECT * FROM labels WHERE trade_key='t1'")
        self.assertEqual(label["price_source"], "historical_trade_print")
        self.assertEqual(label["observed_at"], 2_000_930)
        self.assertEqual(label["entry_price_ppm"], 300_000)

    def test_live_label_retains_book_and_fill_scenarios(self) -> None:
        self._store(trade())
        HistoricalLabeler(self.db, self.api, clock=lambda: 2_000_900).observe_live("t1", 900)
        label = self.db.row("SELECT * FROM labels")
        self.assertEqual(label["status"], "observed_pending_resolution")
        self.assertEqual(label["entry_price_ppm"], 260_000)
        self.assertEqual(label["best_bid_ppm"], 240_000)
        self.assertEqual(label["spread_ppm"], 20_000)
        self.assertGreater(label["ask_depth_5c_microshares"], 0)
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM label_fill_scenarios")["n"], 3)


class PipelineAPI(SampleAPI):
    def __init__(self, raw_market: dict, raw_trade: dict):
        super().__init__([raw_market], {raw_market["conditionId"]: [raw_trade]})
        self.resolution_calls = 0
        self.history_calls = 0

    def resolutions(self, condition_ids: list[str]) -> list[dict]:
        self.resolution_calls += 1
        return [{
            "condition_id": condition_ids[0], "status": "resolved",
            "payouts": [1_000_000, 0], "resolved_at": 2_100_000,
            "was_disputed": False, "resolution_source": "reported",
        }]

    def price_history(self, asset_id: str, start: int, end: int, *, fidelity: int = 1):
        self.history_calls += 1
        points = [
            {"t": 2_000_960, "p": 0.40},
            {"t": 2_003_660, "p": 0.45},
            {"t": 2_086_460, "p": 0.50},
        ]
        return [row for row in points if start <= row["t"] <= end]


class AutomatedPipelineTests(unittest.TestCase):
    def test_sampling_only_pipeline_streams_without_downstream_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampling-only.sqlite3"
            raw_market = sampled_market(
                "condition-sample", "Will the sample happen?", "Elections", "event-sample",
            )
            raw_trade = trade(
                "sample-trade", condition="condition-sample", asset="asset-sample",
                timestamp=2_000_000, title=raw_market["question"], slug="condition-sample",
            )
            api = PipelineAPI(raw_market, raw_trade)
            config = SampledBackfillConfig(
                "sampling-only", days=None, markets_per_category=1, admission_rate=1.0,
            )
            pipeline = AutomatedPipeline(
                path, config, market_workers=2, enrichment_workers=0,
                resolution_workers=0, label_workers=0, downstream=False,
                clock=lambda: 2_000_100, api_factory=lambda: api,
                embedder=FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]}),
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            self.assertEqual(db.row("SELECT phase FROM backfill_cohorts")[0], "complete")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
            self.assertEqual(db.row("SELECT COUNT(*) n FROM labels")["n"], 0)
            self.assertEqual(
                db.row("SELECT COUNT(*) n FROM jobs WHERE job_kind!='scan_market'")["n"], 0,
            )
            self.assertEqual(api.trade_requests, ["condition-sample"])
            db.close()

    def test_lifetime_pipeline_selects_resolves_and_scans_while_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lifetime-pipeline.sqlite3"
            raw_market = sampled_market(
                "condition-lifetime", "Will Company report results?", "Business", "event-life",
            )
            raw_trade = trade(
                "lifetime-trade", condition="condition-lifetime",
                asset="asset-lifetime", timestamp=2_000_000,
                title=raw_market["question"], slug="condition-lifetime",
            )
            api = PipelineAPI(raw_market, raw_trade)
            config = SampledBackfillConfig(
                "lifetime-resolved", days=None, markets_per_category=1,
                admission_rate=1.0, resolution_required=True,
            )
            pipeline = AutomatedPipeline(
                path, config, market_workers=2, enrichment_workers=1,
                resolution_workers=2, label_workers=1, clock=lambda: 2_000_100,
                api_factory=lambda: api,
                embedder=FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]}),
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            cohort = db.row("SELECT * FROM backfill_cohorts")
            self.assertEqual((cohort["history_mode"], cohort["window_start"]), ("lifetime", 1))
            self.assertEqual(cohort["discovery_stop_reason"], "gamma_exhausted")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
            self.assertEqual(db.row("SELECT COUNT(*) n FROM labels")["n"], 3)
            db.close()

    def test_resolution_jobs_are_batched_at_twenty_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batches.sqlite3"
            rows = [
                sampled_market(
                    f"condition-{index}", f"Finance question {index}", "Business", f"event-{index}",
                )
                for index in range(21)
            ]
            api = SampleAPI(rows)
            db = Database(path)
            db.initialize()
            config = SampledBackfillConfig("batched", days=1, resolution_required=True)
            sampled = SampledBackfill(
                db, Collector(db, api, clock=lambda: 2_000_100), config,
                clock=lambda: 2_000_100, embedder=FakeEmbedder({}),
            )
            sampled.discover(sampled._cohort())
            pipeline = AutomatedPipeline(path, config, clock=lambda: 2_000_100)
            pipeline._enqueue_resolution_verification(db)
            jobs = db.rows("SELECT payload_json FROM jobs ORDER BY job_id")
            import json
            self.assertEqual([len(json.loads(row["payload_json"])["conditions"]) for row in jobs], [20, 1])
            db.close()

    def test_concurrent_pipeline_is_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.sqlite3"
            raw_market = sampled_market(
                "condition-1", "Will Company announce a product?", "Business", "event-1",
            )
            raw_trade = trade(size="2400")
            api = PipelineAPI(raw_market, raw_trade)
            embedder = FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]})
            config = SampledBackfillConfig(
                "balanced-resolved", days=1, markets_per_category=1,
                resolution_required=True,
            )
            pipeline = AutomatedPipeline(
                path, config, market_workers=2, enrichment_workers=2,
                resolution_workers=2, label_workers=2, clock=lambda: 2_000_100,
                api_factory=lambda: api, embedder=embedder,
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            self.assertEqual(db.row("SELECT phase FROM backfill_cohorts")["phase"], "complete")
            self.assertEqual(db.row("SELECT resolution_status FROM backfill_cohorts")[0], "complete")
            self.assertEqual(db.row("SELECT enrichment_status FROM trades")[0], "complete")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM labels")["n"], 3)
            self.assertEqual(
                db.row("SELECT COUNT(*) n FROM labels WHERE status='complete'")["n"], 3,
            )
            self.assertEqual(
                db.row("SELECT COUNT(*) n FROM jobs WHERE status='succeeded'")["n"], 6,
            )
            self.assertEqual(api.history_calls, 2)
            JobQueue(db, clock=lambda: 2_000_100).enqueue(
                "observe_entry", "live-future-test",
                {"trade_key": "t1", "horizon": 900, "live": True},
                cohort="balanced-resolved:live", available_at=9_999_999,
            )
            db.close()

            second = AutomatedPipeline(
                path, config, market_workers=2, enrichment_workers=2,
                resolution_workers=2, label_workers=2, clock=lambda: 2_000_100,
                api_factory=lambda: api, embedder=embedder,
            )
            self.assertEqual(second.run(), 0)
            self.assertEqual(api.resolution_calls, 1)
            self.assertEqual(api.history_calls, 2)


class MigrationTests(unittest.TestCase):
    def test_v4_tables_receive_additive_pipeline_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE labels (
                    trade_key TEXT NOT NULL,
                    horizon_seconds INTEGER NOT NULL,
                    scheduled_at INTEGER NOT NULL,
                    observed_at INTEGER,
                    entry_price_ppm INTEGER,
                    price_quality TEXT,
                    outcome_won INTEGER,
                    fee_microusd INTEGER,
                    pnl_microusd INTEGER,
                    PRIMARY KEY(trade_key,horizon_seconds)
                );
                CREATE TABLE backfill_cohorts (
                    cohort_name TEXT PRIMARY KEY, window_start INTEGER NOT NULL,
                    window_end INTEGER NOT NULL, days INTEGER NOT NULL,
                    category_limit INTEGER NOT NULL, seed INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL, similarity_threshold_ppm INTEGER NOT NULL,
                    requested_device TEXT NOT NULL, effective_device TEXT,
                    embedding_device_log TEXT, phase TEXT NOT NULL,
                    discovery_cursor TEXT, discovery_complete INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                """
            )
            connection.close()
            db = Database(path)
            db.initialize()
            label_columns = {row["name"] for row in db.rows("PRAGMA table_info(labels)")}
            cohort_columns = {row["name"] for row in db.rows("PRAGMA table_info(backfill_cohorts)")}
            self.assertTrue({"status", "gross_pnl_microusd", "source_json", "spread_ppm"} <= label_columns)
            self.assertTrue({
                "resolution_required", "resolution_status", "history_mode",
                "admission_rate_ppm", "discovery_pages", "discovery_stop_reason",
                "discovery_strategy",
            } <= cohort_columns)
            self.assertIsNotNone(db.row("SELECT 1 FROM sqlite_master WHERE name='jobs'"))
            self.assertIsNotNone(db.row(
                "SELECT 1 FROM sqlite_master WHERE name='backfill_cohort_discovery_streams'"
            ))
            db.close()
