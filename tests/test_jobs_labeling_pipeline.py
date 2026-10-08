from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from informed_flow.api import APIError, Page
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

    def resolved_markets(self, condition_id: str) -> list[dict]:
        return [
            row for row in self.resolution_rows
            if row.get("conditionId", row.get("condition_id")) == condition_id
        ]

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

    def test_http_500_job_can_be_audited_as_skipped_without_becoming_dead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "queue.sqlite3")
            db.initialize()
            queue = JobQueue(db, clock=lambda: 100)
            queue.enqueue("scan_market", "scan:a", {"condition": "a"})
            job = queue.claim("worker", ("scan_market",))
            queue.skip(job, "worker", APIError("server error", status=500, retryable=True))
            row = db.row("SELECT status,last_error FROM jobs WHERE job_key='scan:a'")
            self.assertEqual(row["status"], "succeeded")
            self.assertIn("skipped after HTTP 500", row["last_error"])
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
            "conditionId": "condition-1", "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": "[\"Yes\",\"No\"]",
            "outcomePrices": "[\"1\",\"0\"]",
            "clobTokenIds": "[\"asset-1\",\"asset-no\"]",
            "closedTime": resolved_at, "wasDisputed": True,
        }]
        ResolutionStore(self.db, self.api, clock=lambda: 2_200_000).verify(
            "unused", ["condition-1"],
        )

    def test_resolution_accepts_gamma_one_hot_and_rejects_ambiguous_or_pending(self) -> None:
        self.assertEqual(
            resolution_result({
                "closed": True, "outcomes": ["Yes", "No"],
                "outcomePrices": ["1", "0"], "clobTokenIds": ["yes", "no"],
            }),
            (True, 0, "resolved", "terminal_outcome_prices"),
        )
        self.assertEqual(
            resolution_result({
                "closed": True, "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.5", "0.5"], "clobTokenIds": ["yes", "no"],
            })[3],
            "ambiguous_outcome_prices",
        )
        self.assertEqual(
            resolution_result({
                "closed": True, "umaResolutionStatus": "proposed",
                "outcomes": ["Yes", "No"], "outcomePrices": ["1", "0"],
                "clobTokenIds": ["yes", "no"],
            })[3],
            "nonterminal_resolution",
        )
        self.assertEqual(
            resolution_result({
                "closed": False, "outcomes": ["Yes", "No"],
                "outcomePrices": ["1", "0"], "clobTokenIds": ["yes", "no"],
            })[3],
            "not_closed",
        )

    def test_gamma_resolution_validation_rejects_malformed_or_ambiguous_rows(self) -> None:
        valid = {
            "conditionId": "condition-1", "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": ["Yes", "No"], "outcomePrices": ["1", "0"],
            "clobTokenIds": ["yes", "no"],
        }
        variants = {
            "condition_mismatch": {"conditionId": "other"},
            "missing_outcomes": {"outcomes": None},
            "outcome_length_mismatch": {"clobTokenIds": ["yes"]},
            "duplicate_outcomes": {"outcomes": ["Yes", "yes"]},
            "duplicate_clob_tokens": {"clobTokenIds": ["same", "same"]},
            "ambiguous_outcome_prices": {"outcomePrices": ["1", "1"]},
            "invalid_outcome_prices": {"outcomePrices": ["NaN", "0"]},
            "nonterminal_resolution": {"umaResolutionStatus": "disputed"},
        }
        from informed_flow.labeling import gamma_resolution_evidence
        for expected, changes in variants.items():
            with self.subTest(expected=expected):
                row = dict(valid)
                row.update(changes)
                self.assertEqual(
                    gamma_resolution_evidence(row, "condition-1").quality, expected,
                )

    def test_deepseek_shape_maps_no_token_to_winner_index_one(self) -> None:
        raw = {
            "conditionId": "0xee17b7", "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": "[\"Yes\",\"No\"]",
            "outcomePrices": "[\"0\",\"1\"]",
            "clobTokenIds": "[\"yes-token\",\"no-token\"]",
        }
        self.assertEqual(
            resolution_result(raw),
            (True, 1, "resolved", "terminal_outcome_prices"),
        )

    def test_resolution_timestamp_uses_closed_time_only(self) -> None:
        row = sampled_market(
            "condition-1", "Will timestamp be exact?", "Business", "event-time",
        )
        row.pop("closedTime")
        row["endDate"] = "2026-12-31T00:00:00Z"
        row["umaEndDate"] = "2026-12-30T00:00:00Z"
        self.api.resolution_rows = [row]
        ResolutionStore(self.db, self.api, clock=lambda: 2_200_000).verify(
            "unused", ["condition-1"],
        )
        resolution = self.db.row("SELECT resolved_at,source FROM market_resolutions")
        self.assertIsNone(resolution["resolved_at"])
        self.assertEqual(resolution["source"], "gamma_markets_v1")

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
        terminal = dict(rows[0])
        pending = dict(rows[1])
        pending["umaResolutionStatus"] = "proposed"
        self.api.resolution_rows = [terminal, pending]
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

    def test_gamma_token_map_repairs_unknown_trade_outcome_index(self) -> None:
        self.api.resolution_rows = [{
            "conditionId": "condition-1", "closed": True,
            "outcomes": ["Yes", "No"], "outcomePrices": ["0", "1"],
            "clobTokenIds": ["asset-yes", "asset-no"], "closedTime": 2_100_000,
        }]
        ResolutionStore(self.db, self.api, clock=lambda: 2_200_000).verify(
            "unused", ["condition-1"],
        )
        raw_trade = trade("unknown-index", asset="asset-no")
        raw_trade["outcome"] = "No"
        raw_trade["outcome_index"] = 999
        self._store(raw_trade)
        conflicting_trade = trade(
            "conflicting-index", asset="asset-no", timestamp=2_000_001,
        )
        conflicting_trade["outcome"] = "No"
        conflicting_trade["outcome_index"] = 0
        self._store(conflicting_trade)
        self.api.history_rows = [{"t": 2_000_960, "p": 0.40}]
        HistoricalLabeler(self.db, self.api).observe("cohort", "unknown-index", 900)
        outcome = self.db.row("SELECT outcome_index FROM outcomes WHERE asset_id='asset-no'")
        label = self.db.row("SELECT * FROM labels WHERE trade_key='unknown-index'")
        self.assertEqual(outcome["outcome_index"], 1)
        self.assertEqual((label["outcome_won"], label["gross_pnl_microusd"]), (1, 600_000))

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

    def resolved_markets(self, condition_id: str) -> list[dict]:
        self.resolution_calls += 1
        row = dict(self.markets_by_condition[condition_id])
        row["clobTokenIds"] = [self.trades[condition_id][0]["asset_id"], f"no-{condition_id}"]
        row["closedTime"] = 2_100_000
        return [row]

    def price_history(self, asset_id: str, start: int, end: int, *, fidelity: int = 1):
        self.history_calls += 1
        points = [
            {"t": 2_000_960, "p": 0.40},
            {"t": 2_003_660, "p": 0.45},
            {"t": 2_086_460, "p": 0.50},
        ]
        return [row for row in points if start <= row["t"] <= end]


class AutomatedPipelineTests(unittest.TestCase):
    def test_finish_existing_refuses_legacy_cohort_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-finish.sqlite3"
            db = Database(path)
            db.initialize()
            config = SampledBackfillConfig("legacy", days=None, resolution_required=False)
            SampledBackfill(
                db, Collector(db, SampleAPI([])), config, embedder=FakeEmbedder({}),
            )._cohort()
            before = dict(db.row("SELECT * FROM backfill_cohorts WHERE cohort_name='legacy'"))
            db.close()
            pipeline = AutomatedPipeline(path, SampledBackfillConfig("legacy"))
            self.assertEqual(pipeline.finish_existing(), 1)
            db = Database(path)
            db.initialize()
            after = dict(db.row("SELECT * FROM backfill_cohorts WHERE cohort_name='legacy'"))
            self.assertEqual(after["discovery_complete"], before["discovery_complete"])
            self.assertEqual(after["phase"], before["phase"])
            db.close()

    def test_finish_existing_freezes_selection_without_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finish-existing.sqlite3"
            raw_market = sampled_market(
                "condition-existing", "Will existing result happen?", "Business", "event-existing",
            )
            raw_trade = trade(
                "existing-trade", condition="condition-existing", asset="asset-existing",
                timestamp=2_000_000, title=raw_market["question"], slug="condition-existing",
            )
            api = PipelineAPI(raw_market, raw_trade)
            config = SampledBackfillConfig(
                "finish-existing", days=None, markets_per_category=10,
                resolution_required=True,
            )
            db = Database(path)
            db.initialize()
            sampled = SampledBackfill(
                db, Collector(db, api, clock=lambda: 2_000_100), config,
                clock=lambda: 2_000_100, embedder=FakeEmbedder({}),
            )
            sampled._cohort()
            sampled._discover_market(raw_market, 1, 2_000_100, 1)
            with db.connection:
                db.connection.execute(
                    """UPDATE backfill_cohort_markets
                       SET selection_status='selected',fetch_status='pending'
                       WHERE cohort_name=? AND condition_id=?""",
                    (config.cohort, "condition-existing"),
                )
            ResolutionStore(db, api, clock=lambda: 2_000_100).verify(
                config.cohort, ["condition-existing"],
            )
            requests_before = len(api.market_requests)
            resolution_requests_before = api.resolution_calls
            db.close()

            pipeline = AutomatedPipeline(
                path, config, market_workers=1, enrichment_workers=1,
                resolution_workers=1, label_workers=1, clock=lambda: 2_000_100,
                api_factory=lambda: api,
            )
            self.assertEqual(pipeline.finish_existing(), 0)
            self.assertEqual(len(api.market_requests), requests_before)
            self.assertEqual(api.resolution_calls, resolution_requests_before)
            db = Database(path)
            db.initialize()
            cohort = db.row("SELECT * FROM backfill_cohorts WHERE cohort_name='finish-existing'")
            self.assertEqual((cohort["discovery_complete"], cohort["phase"]), (1, "complete"))
            self.assertEqual(cohort["discovery_stop_reason"], "finish_existing")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
            self.assertEqual(db.row("SELECT COUNT(*) n FROM labels")["n"], 3)
            db.close()

    def test_http_500_market_is_skipped_without_stopping_other_market_scans(self) -> None:
        class OneBrokenMarketAPI(SampleAPI):
            def trades_page(self, **kwargs):
                condition = kwargs["condition"]
                self.trade_requests.append(condition)
                if condition == "condition-broken":
                    raise APIError("temporary server failure", status=500, retryable=True)
                return Page(self.trades.get(condition, []), None)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skip-500.sqlite3"
            broken = sampled_market(
                "condition-broken", "Will broken result happen?", "Business", "event-broken",
            )
            good = sampled_market(
                "condition-good", "Will good result happen?", "Business", "event-good",
            )
            raw_trade = trade(
                "good-trade", condition="condition-good", asset="asset-good",
                timestamp=2_000_000, title=good["question"], slug="condition-good",
            )
            api = OneBrokenMarketAPI([broken, good], {"condition-good": [raw_trade]})
            vectors = {
                normalize_title(broken["question"]): [1.0, 0.0],
                normalize_title(good["question"]): [0.0, 1.0],
            }
            pipeline = AutomatedPipeline(
                path,
                SampledBackfillConfig(
                    "skip-500", days=None, markets_per_category=2, admission_rate=1.0,
                    resolution_required=True,
                ),
                market_workers=1, enrichment_workers=0, resolution_workers=1,
                label_workers=0, downstream=False, clock=lambda: 2_000_100,
                api_factory=lambda: api, embedder=FakeEmbedder(vectors),
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            self.assertEqual(db.row("SELECT phase FROM backfill_cohorts")["phase"], "complete")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
            broken_row = db.row(
                "SELECT fetch_status,last_error FROM backfill_cohort_markets WHERE condition_id='condition-broken'"
            )
            self.assertEqual(broken_row["fetch_status"], "failed")
            self.assertIsNotNone(broken_row["last_error"])
            skipped = db.row(
                """SELECT status,last_error FROM jobs
                   WHERE entity_id='condition-broken' AND job_kind='scan_market'"""
            )
            self.assertEqual(skipped["status"], "succeeded")
            self.assertIn("skipped after HTTP 500", skipped["last_error"])
            db.close()

    def test_http_500_resolution_isolated_to_one_condition(self) -> None:
        class OneBrokenResolutionAPI(SampleAPI):
            def resolved_markets(self, condition_id):
                if condition_id == "condition-broken":
                    raise APIError("Gamma resolution failed", status=500, retryable=True)
                return super().resolved_markets(condition_id)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolution-500.sqlite3"
            broken = sampled_market(
                "condition-broken", "Will broken resolve?", "Business", "event-broken",
            )
            good = sampled_market(
                "condition-good", "Will good resolve?", "Business", "event-good",
            )
            good_trade = trade(
                "good-trade", condition="condition-good", asset="asset-condition-good",
                timestamp=2_000_000, title=good["question"], slug="condition-good",
            )
            api = OneBrokenResolutionAPI([broken, good], {"condition-good": [good_trade]})
            vectors = {
                normalize_title(broken["question"]): [1.0, 0.0],
                normalize_title(good["question"]): [0.0, 1.0],
            }
            pipeline = AutomatedPipeline(
                path,
                SampledBackfillConfig(
                    "resolution-500", days=None, markets_per_category=2,
                    admission_rate=1.0, resolution_required=True,
                ),
                market_workers=1, enrichment_workers=0, resolution_workers=1,
                label_workers=0, downstream=False, clock=lambda: 2_000_100,
                api_factory=lambda: api, embedder=FakeEmbedder(vectors),
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            broken_row = db.row(
                """SELECT eligible,selection_status,eligibility_reason
                   FROM backfill_cohort_markets WHERE condition_id='condition-broken'"""
            )
            good_row = db.row(
                """SELECT selection_status FROM backfill_cohort_markets
                   WHERE condition_id='condition-good'"""
            )
            self.assertEqual(
                tuple(broken_row), (0, "ineligible", "resolution_http_500"),
            )
            self.assertEqual(good_row["selection_status"], "selected")
            self.assertEqual(db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
            skipped = db.row(
                """SELECT status,last_error FROM jobs
                   WHERE job_kind='verify_resolution' AND entity_id='condition-broken'"""
            )
            self.assertEqual(skipped["status"], "succeeded")
            self.assertIn("skipped after HTTP 500", skipped["last_error"])
            db.close()

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
                resolution_required=True,
            )
            pipeline = AutomatedPipeline(
                path, config, market_workers=2, enrichment_workers=0,
                resolution_workers=1, label_workers=0, downstream=False,
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
                db.row(
                    """SELECT COUNT(*) n FROM jobs
                       WHERE job_kind NOT IN ('scan_market','verify_resolution')"""
                )["n"], 0,
            )
            self.assertEqual(api.trade_requests, ["condition-sample"])
            db.close()

    def test_stale_scan_enrichment_and_label_jobs_are_audited_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale-jobs.sqlite3"
            raw_market = sampled_market(
                "condition-stale", "Will stale work run?", "Business", "event-stale",
            )
            raw_trade = trade(
                "stale-trade", condition="condition-stale", asset="asset-condition-stale",
                size="5000", timestamp=2_000_000, title=raw_market["question"],
                slug="condition-stale",
            )
            api = PipelineAPI(raw_market, raw_trade)
            config = SampledBackfillConfig(
                "stale", days=None, markets_per_category=1, admission_rate=1.0,
                resolution_required=True,
            )
            pipeline = AutomatedPipeline(
                path, config, market_workers=1, enrichment_workers=1,
                resolution_workers=1, label_workers=1, downstream=False,
                clock=lambda: 2_000_100, api_factory=lambda: api,
                embedder=FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]}),
            )
            self.assertEqual(pipeline.run(), 0)
            db = Database(path)
            db.initialize()
            pipeline._enqueue_scan(db, "condition-stale", reopen_succeeded=True)
            pipeline._enqueue_trade_jobs(db, "stale")
            with db.connection:
                db.connection.execute(
                    """UPDATE backfill_cohort_markets
                       SET eligible=0,selection_status='ineligible',fetch_status='not_selected'
                       WHERE cohort_name='stale' AND condition_id='condition-stale'"""
                )
            requests_before = len(api.trade_requests)
            self.assertEqual(pipeline._historical_workers(db), 0)
            self.assertEqual(len(api.trade_requests), requests_before)
            self.assertEqual(api.wallet_history_calls, 0)
            self.assertEqual(api.history_calls, 0)
            stale_jobs = db.rows(
                """SELECT job_kind,last_error FROM jobs
                   WHERE job_kind IN ('scan_market','enrich_trade','observe_entry')"""
            )
            self.assertTrue(stale_jobs)
            self.assertTrue(all(
                row["last_error"] and "skipped as stale/ineligible" in row["last_error"]
                for row in stale_jobs
            ))
            db.close()

    def test_data_resolution_row_does_not_satisfy_gamma_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-resolution.sqlite3"
            raw_market = sampled_market(
                "condition-data", "Will Data be ignored?", "Business", "event-data",
            )
            api = SampleAPI([raw_market])
            config = SampledBackfillConfig("gamma-cache", days=1, resolution_required=True)
            db = Database(path)
            db.initialize()
            sampled = SampledBackfill(
                db, Collector(db, api, clock=lambda: 2_000_100), config,
                clock=lambda: 2_000_100, embedder=FakeEmbedder({}),
            )
            sampled.discover(sampled._cohort())
            with db.connection:
                db.connection.execute(
                    """INSERT INTO market_resolutions(
                           condition_id,status,terminal,winning_outcome_index,payouts_json,
                           resolved_at,was_disputed,source,quality,raw_json,checked_at
                       ) VALUES (?,?,1,0,?,NULL,0,'data_v2_resolutions',?,?,?)""",
                    ("condition-data", "resolved", "[1000000,0]", "terminal_payout", "{}", 1),
                )
            AutomatedPipeline(path, config)._enqueue_resolution_verification(db)
            job = db.row("SELECT * FROM jobs WHERE job_kind='verify_resolution'")
            self.assertEqual(job["entity_id"], "condition-data")
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

    def test_resolution_jobs_are_isolated_per_condition(self) -> None:
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
            self.assertEqual(len(jobs), 21)
            self.assertEqual(
                [json.loads(row["payload_json"])["condition_id"] for row in jobs],
                sorted(row["conditionId"] for row in rows),
            )
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
                "discovery_strategy", "resolution_source",
            } <= cohort_columns)
            self.assertIsNotNone(db.row("SELECT 1 FROM sqlite_master WHERE name='jobs'"))
            self.assertIsNotNone(db.row(
                "SELECT 1 FROM sqlite_master WHERE name='backfill_cohort_discovery_streams'"
            ))
            db.close()
