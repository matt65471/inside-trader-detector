from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from informed_flow.api import APIError, Page
from informed_flow.cli import _export, _report
from informed_flow.db import Database
from informed_flow.sampled_backfill import (
    CATEGORY_TAG_SLUGS,
    LocalSemanticEmbedder,
    SampledBackfill,
    SampledBackfillConfig,
    normalize_title,
)
from informed_flow.service import Collector, Counts
from tests.test_service import FakeAPI, market, trade


def sampled_market(
    condition: str,
    title: str,
    category: str,
    event_id: str,
    *,
    event_slug: str | None = None,
) -> dict:
    row = market(condition)
    row.update({
        "id": f"market-{condition}",
        "conditionId": condition,
        "question": title,
        "slug": condition,
        "closed": True,
        "createdAt": "1970-01-20T00:00:00Z",
        "endDate": "1970-01-30T00:00:00Z",
        "events": [{
            "id": event_id,
            "slug": event_slug or event_id,
            "category": category,
        }],
    })
    return row


class SampleAPI(FakeAPI):
    def __init__(self, markets: list[dict], trades: dict[str, list[dict]] | None = None):
        super().__init__()
        self.discovery_pages: list[Page] | None = None
        self.tag_ids = {
            slug: str(index + 1)
            for index, slug in enumerate(
                slug for slugs in CATEGORY_TAG_SLUGS.values() for slug in slugs
            )
        }
        category_slug = {
            "sports": "sports", "crypto": "crypto", "weather": "weather",
            "culture": "pop-culture", "pop-culture": "pop-culture",
            "finance": "finance", "economy": "economy", "business": "business",
            "geopolitics": "geopolitics", "world": "world",
            "politics": "politics", "elections": "elections",
        }
        self.markets_by_tag = {tag_id: [] for tag_id in self.tag_ids.values()}
        for row in markets:
            events = row.get("events") or []
            category = str((events[0] if events else {}).get("category", "")).lower()
            slug = category_slug.get(category)
            if slug:
                self.markets_by_tag[self.tag_ids[slug]].append(row)
        self.market_requests: list[tuple[str | None, bool, str | None]] = []
        self.trade_requests: list[str] = []
        self.trades = trades or {}

    def tag_by_slug(self, slug):
        return {"id": self.tag_ids[slug], "slug": slug}

    def markets_page(self, cursor=None, *, closed=False, tag_id=None):
        self.market_requests.append((cursor, closed, str(tag_id) if tag_id is not None else None))
        if self.discovery_pages is not None and str(tag_id) == self.tag_ids["sports"]:
            return self.discovery_pages.pop(0)
        return Page(self.markets_by_tag[str(tag_id)], None)

    def trades_page(self, **kwargs):
        condition = kwargs["condition"]
        self.trade_requests.append(condition)
        return Page(self.trades.get(condition, []), None)


class FakeEmbedder:
    failures: list[str] = []

    def __init__(self, vectors: dict[str, list[float]], device: str = "cpu"):
        self.vectors = vectors
        self.device = device
        self.calls = 0

    def iter_batches(self, texts):
        self.calls += 1
        yield self.device, [self.vectors[text] for text in texts]


class SampledBackfillTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tempdir.name) / "sample.sqlite3")
        self.db.initialize()

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def test_discovers_deduplicates_selects_and_keeps_zero_trade_markets(self):
        rows = [
            sampled_market("a", "Will Alpha win the award?", "Business", "event-alpha"),
            sampled_market("b", "Does Alpha win this award?", "Business", "event-alpha"),
            sampled_market("c", "Will Alpha receive the award?", "Business", "event-charlie"),
            sampled_market("d", "Will rates be cut this year?", "Business", "event-delta"),
        ]
        vectors = {
            normalize_title(rows[0]["question"]): [1.0, 0.0],
            normalize_title(rows[1]["question"]): [1.0, 0.0],
            normalize_title(rows[2]["question"]): [0.95, 0.3122499],
            normalize_title(rows[3]["question"]): [0.0, 1.0],
        }
        api = SampleAPI(rows)
        embedder = FakeEmbedder(vectors, "mps")
        runner = SampledBackfill(
            self.db,
            Collector(self.db, api, clock=lambda: 2_000_100),
            SampledBackfillConfig("balanced", days=1, markets_per_category=10),
            clock=lambda: 2_000_100,
            embedder=embedder,
        )
        self.assertEqual(runner.run(), 0)
        self.assertEqual(len(api.market_requests), 11)
        self.assertTrue(all(closed and tag_id for _, closed, tag_id in api.market_requests))
        self.assertEqual(
            self.db.row("SELECT phase FROM backfill_cohorts WHERE cohort_name='balanced'")[0],
            "complete",
        )
        statuses = {
            row["selection_status"]: row["n"]
            for row in self.db.rows(
                """SELECT selection_status,COUNT(*) n FROM backfill_cohort_markets
                   GROUP BY selection_status"""
            )
        }
        self.assertEqual(statuses, {"redundant": 2, "selected": 2})
        self.assertEqual(
            self.db.row("SELECT COUNT(*) FROM backfill_cohort_markets WHERE fetch_status='complete'")[0],
            2,
        )
        self.assertEqual(
            self.db.row("SELECT SUM(qualifying_trade_count) FROM backfill_cohort_markets")[0], 0
        )
        scores = [
            row[0] for row in self.db.rows(
                "SELECT similarity_ppm FROM backfill_cohort_markets WHERE selection_status='redundant'"
            )
        ]
        self.assertIn(1_000_000, scores)
        self.assertTrue(any(900_000 <= score < 1_000_000 for score in scores))
        reasons = {
            row[0] for row in self.db.rows(
                "SELECT selection_reason FROM backfill_cohort_markets WHERE selection_status='redundant'"
            )
        }
        self.assertTrue("semantic_similarity" in reasons)
        self.assertTrue(reasons & {"same_event_id", "same_event_slug"})
        self.assertEqual(
            self.db.row("SELECT effective_device FROM backfill_cohorts")[0], "mps"
        )
        calls = embedder.calls
        self.assertEqual(runner.run(), 0)
        self.assertEqual(embedder.calls, calls)

    def test_reuses_existing_trade_and_exports_only_selected_cohort(self):
        raw_market = sampled_market("p1", "Will the bill pass?", "Elections", "event-politics")
        raw_trade = trade(
            "existing", condition="p1", asset="asset-p1", timestamp=2_000_000,
            title=raw_market["question"], slug="p1",
        )
        api = SampleAPI([raw_market], {"p1": [raw_trade]})
        collector = Collector(self.db, api, clock=lambda: 2_000_100)
        collector.process_trade(
            raw_trade, "backfill", self.db.start_run("backfill", 2_000_100), Counts(),
            market_raw=raw_market,
        )
        runner = SampledBackfill(
            self.db, collector, SampledBackfillConfig("politics", days=1),
            clock=lambda: 2_000_100,
            embedder=FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]}),
        )
        self.assertEqual(runner.run(), 0)
        self.assertEqual(self.db.row("SELECT COUNT(*) FROM trades")[0], 1)
        selected = self.db.row("SELECT * FROM backfill_cohort_markets")
        self.assertEqual(selected["canonical_category"], "politics")
        self.assertEqual(selected["qualifying_trade_count"], 1)

        output = Path(self.tempdir.name) / "exports"
        _export(self.db, output, "politics")
        for name in (
            "cohort_markets.csv", "cohort_trades.csv", "cohort_similarity_rejections.csv",
        ):
            self.assertTrue((output / name).exists())
        with (output / "cohort_trades.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 2)
        _report(self.db, 1, "politics")

    def test_inherits_legacy_window_and_rejects_configuration_changes(self):
        self.db.save_checkpoint(
            "market_backfill_90d", cursor=None, window_start=100, window_end=200,
            last_trade_ts=None,
        )
        api = SampleAPI([])
        collector = Collector(self.db, api, clock=lambda: 500)
        runner = SampledBackfill(
            self.db, collector, SampledBackfillConfig("frozen"), clock=lambda: 500,
            embedder=FakeEmbedder({}),
        )
        cohort = runner._cohort()
        self.assertEqual((cohort["window_start"], cohort["window_end"]), (100, 200))
        changed = SampledBackfill(
            self.db, collector, SampledBackfillConfig("frozen", seed=1), clock=lambda: 500,
            embedder=FakeEmbedder({}),
        )
        with self.assertRaisesRegex(ValueError, "different configuration"):
            changed._cohort()
        resolution_changed = SampledBackfill(
            self.db, collector,
            SampledBackfillConfig("frozen", resolution_required=True),
            clock=lambda: 500, embedder=FakeEmbedder({}),
        )
        with self.assertRaisesRegex(ValueError, "resolution_required"):
            resolution_changed._cohort()
        with self.db.connection:
            self.db.connection.execute(
                "UPDATE backfill_cohorts SET discovery_strategy='global_v1' WHERE cohort_name='frozen'"
            )
        with self.assertRaisesRegex(ValueError, "discovery_strategy"):
            runner._cohort()

    def test_overlapping_tag_results_are_stored_once(self):
        raw = sampled_market(
            "overlap", "Will rates change?", "Finance", "event-overlap",
        )
        api = SampleAPI([raw])
        api.markets_by_tag[api.tag_ids["economy"]] = [raw]
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 2_000_100),
            SampledBackfillConfig("overlap", days=1), clock=lambda: 2_000_100,
            embedder=FakeEmbedder({}),
        )
        runner.discover(runner._cohort())
        self.assertEqual(
            self.db.row("SELECT COUNT(*) n FROM backfill_cohort_markets")["n"], 1,
        )

    def test_http_500_discovery_stream_is_skipped_and_other_tags_continue(self):
        class BrokenSportsAPI(SampleAPI):
            def markets_page(self, cursor=None, *, closed=False, tag_id=None):
                if str(tag_id) == self.tag_ids["sports"]:
                    raise APIError("Gamma market feed failed", status=500, retryable=True)
                return super().markets_page(cursor, closed=closed, tag_id=tag_id)

        raw = sampled_market(
            "finance-ok", "Will the company announce results?", "Business", "event-ok",
        )
        api = BrokenSportsAPI([raw])
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 2_000_100),
            SampledBackfillConfig(
                "skip-discovery-500", days=None, markets_per_category=1,
                admission_rate=1.0,
            ),
            clock=lambda: 2_000_100,
            embedder=FakeEmbedder({normalize_title(raw["question"]): [1.0, 0.0]}),
        )
        self.assertEqual(runner.run(), 0)
        stream = self.db.row(
            """SELECT exhausted,last_error FROM backfill_cohort_discovery_streams
               WHERE cohort_name='skip-discovery-500' AND tag_slug='sports'"""
        )
        self.assertEqual(stream["exhausted"], 1)
        self.assertIn("Gamma market feed failed", stream["last_error"])
        self.assertEqual(
            self.db.row(
                """SELECT selection_status FROM backfill_cohort_markets
                   WHERE cohort_name='skip-discovery-500' AND condition_id='finance-ok'"""
            )["selection_status"],
            "selected",
        )
        self.assertEqual(self.db.row("SELECT phase FROM backfill_cohorts")["phase"], "complete")

    def test_discovery_resumes_after_repeated_cursor(self):
        first_api = SampleAPI([])
        first_api.discovery_pages = [Page([], "same"), Page([], "same")]
        config = SampledBackfillConfig("resume", days=1)
        first = SampledBackfill(
            self.db, Collector(self.db, first_api, clock=lambda: 2_000_100), config,
            clock=lambda: 2_000_100, embedder=FakeEmbedder({}),
        )
        self.assertEqual(first.run(), 1)
        failed = self.db.row("SELECT * FROM backfill_cohorts WHERE cohort_name='resume'")
        self.assertEqual(failed["phase"], "discovering")
        stream = self.db.row(
            "SELECT * FROM backfill_cohort_discovery_streams WHERE tag_slug='sports'"
        )
        self.assertEqual(stream["cursor"], "same")

        resumed_api = SampleAPI([])
        resumed = SampledBackfill(
            self.db, Collector(self.db, resumed_api, clock=lambda: 2_000_200), config,
            clock=lambda: 2_000_200, embedder=FakeEmbedder({}),
        )
        self.assertEqual(resumed.run(), 0)
        self.assertEqual(resumed_api.market_requests[0], ("same", True, "1"))
        self.assertEqual(self.db.row("SELECT phase FROM backfill_cohorts")[0], "complete")

    def test_category_aliases_and_window_eligibility_are_persisted(self):
        rows = [
            sampled_market("e", "Economy question", "Economy", "event-e"),
            sampled_market("w", "World question", "World", "event-w"),
            sampled_market("p", "Election question", "Elections", "event-p"),
            sampled_market("old", "Old finance question", "Finance", "event-old"),
            sampled_market("future", "Future finance question", "Finance", "event-future"),
        ]
        rows[3]["endDate"] = "1970-01-01T00:00:01Z"
        rows[4]["createdAt"] = "1970-02-01T00:00:00Z"
        api = SampleAPI(rows)
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 2_000_100),
            SampledBackfillConfig("categories", days=1), clock=lambda: 2_000_100,
            embedder=FakeEmbedder({}),
        )
        cohort = runner._cohort()
        runner.discover(cohort)
        categories = {
            row["condition_id"]: (row["canonical_category"], row["selection_status"], row["eligibility_reason"])
            for row in self.db.rows("SELECT * FROM backfill_cohort_markets")
        }
        self.assertEqual(categories["e"][:2], ("finance", "candidate"))
        self.assertEqual(categories["w"][:2], ("geopolitics", "candidate"))
        self.assertEqual(categories["p"][:2], ("politics", "candidate"))
        self.assertEqual(categories["old"][2], "ended_before_window")
        self.assertEqual(categories["future"][2], "created_after_window")

    def test_lifetime_streaming_stops_as_soon_as_every_category_is_full(self):
        rows = [
            sampled_market(
                f"condition-{index}", f"Unique question {index}", category, f"event-{index}",
            )
            for index, category in enumerate(
                ("Sports", "Crypto", "Weather", "Culture", "Finance", "World", "Elections")
            )
        ]
        api = SampleAPI([])
        api.discovery_pages = [Page(rows, "unused-next-page"), Page([], None)]
        vectors = {
            normalize_title(row["question"]): [float(position == index) for position in range(7)]
            for index, row in enumerate(rows)
        }
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 10_000_000),
            SampledBackfillConfig(
                "lifetime-stop", days=None, markets_per_category=1, admission_rate=1.0,
            ),
            clock=lambda: 10_000_000, embedder=FakeEmbedder(vectors),
        )
        runner.stream_select()
        self.assertEqual(api.market_requests, [(None, True, "1")])
        cohort = self.db.row("SELECT * FROM backfill_cohorts")
        self.assertEqual(cohort["history_mode"], "lifetime")
        self.assertEqual(cohort["window_start"], 1)
        self.assertEqual(cohort["discovery_stop_reason"], "quotas_filled")
        self.assertEqual(
            self.db.row(
                "SELECT COUNT(*) n FROM backfill_cohort_markets WHERE selection_status='selected'"
            )["n"],
            7,
        )

    def test_lifetime_streaming_logs_and_skips_missing_condition_ids(self):
        valid = sampled_market(
            "valid-condition", "Will the valid market resolve?", "Elections", "valid-event",
        )
        missing = dict(valid)
        missing.pop("conditionId")
        missing["id"] = "missing-condition"
        api = SampleAPI([missing, valid])
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 10_000_000),
            SampledBackfillConfig(
                "skip-missing", days=None, markets_per_category=1, admission_rate=1.0,
            ),
            clock=lambda: 10_000_000,
            embedder=FakeEmbedder({normalize_title(valid["question"]): [1.0, 0.0]}),
        )
        runner.stream_select()
        self.assertEqual(
            self.db.row(
                "SELECT COUNT(*) n FROM backfill_cohort_markets WHERE selection_status='selected'"
            )["n"],
            1,
        )
        error = self.db.row(
            "SELECT * FROM collection_errors WHERE stage='sampled_discovery'"
        )
        self.assertIsNotNone(error)
        self.assertIn("missing_condition_id", error["error_message"])

    def test_lifetime_exhaustion_reconsiders_random_rejects_and_semantic_duplicates(self):
        rows = [
            sampled_market("one", "Will Alpha happen?", "Finance", "event-one"),
            sampled_market("duplicate", "Does Alpha happen?", "Finance", "event-two"),
            sampled_market("two", "Will rates fall?", "Finance", "event-three"),
        ]
        vectors = {
            normalize_title(rows[0]["question"]): [1.0, 0.0],
            normalize_title(rows[1]["question"]): [1.0, 0.0],
            normalize_title(rows[2]["question"]): [0.0, 1.0],
        }
        api = SampleAPI(rows)
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 10_000_000),
            SampledBackfillConfig(
                "fallback", days=None, markets_per_category=2,
                admission_rate=0.000000001,
            ),
            clock=lambda: 10_000_000, embedder=FakeEmbedder(vectors),
        )
        runner.stream_select()
        statuses = {
            row["selection_status"]: row["n"] for row in self.db.rows(
                "SELECT selection_status,COUNT(*) n FROM backfill_cohort_markets GROUP BY selection_status"
            )
        }
        self.assertEqual(statuses, {"redundant": 1, "selected": 2})
        self.assertEqual(
            self.db.row("SELECT discovery_stop_reason FROM backfill_cohorts")[0],
            "gamma_exhausted",
        )

    def test_lifetime_fetch_keeps_trades_older_than_ninety_days(self):
        raw_market = sampled_market(
            "old-market", "Will the old event resolve?", "Elections", "old-event",
        )
        old_trade = trade(
            "old-trade", condition="old-market", asset="asset-old", timestamp=100,
            title=raw_market["question"], slug="old-market",
        )
        api = SampleAPI([raw_market], {"old-market": [old_trade]})
        runner = SampledBackfill(
            self.db, Collector(self.db, api, clock=lambda: 10_000_000),
            SampledBackfillConfig(
                "lifetime-trades", days=None, markets_per_category=1, admission_rate=1.0,
            ),
            clock=lambda: 10_000_000,
            embedder=FakeEmbedder({normalize_title(raw_market["question"]): [1.0, 0.0]}),
        )
        self.assertEqual(runner.run(), 0)
        self.assertEqual(self.db.row("SELECT COUNT(*) n FROM trades")["n"], 1)
        self.assertEqual(
            self.db.row("SELECT qualifying_trade_count FROM backfill_cohort_markets")[0], 1,
        )


class Availability:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, *, cuda: bool, mps: bool):
        self.cuda = Availability(cuda)
        self.backends = type("Backends", (), {"mps": Availability(mps)})()


class EncodingModel:
    def __init__(self, device: str, failures: set[str], calls: list[tuple[str, int]]):
        self.device = device
        self.failures = failures
        self.calls = calls

    def encode(self, texts, **kwargs):
        self.calls.append((self.device, len(texts)))
        if self.device in self.failures:
            raise RuntimeError(f"{self.device} failed")
        return [[1.0, 0.0] for _ in texts]


class DeviceSelectionTests(unittest.TestCase):
    def _runner(self, torch, failures=(), requested="auto"):
        loaded = []
        calls = []

        def load(_name, *, device):
            loaded.append(device)
            return EncodingModel(device, set(failures), calls)

        runner = LocalSemanticEmbedder(
            "test-model", requested, torch_module=torch, model_loader=load, batch_size=8,
        )
        return runner, loaded, calls

    def test_cuda_is_preferred(self):
        runner, loaded, _ = self._runner(FakeTorch(cuda=True, mps=True))
        batches = list(runner.iter_batches(["one"]))
        self.assertEqual(loaded, ["cuda"])
        self.assertEqual(batches[0][0], "cuda")

    def test_cuda_failure_falls_back_to_mps(self):
        runner, loaded, _ = self._runner(FakeTorch(cuda=True, mps=True), {"cuda"})
        batches = list(runner.iter_batches(["one"]))
        self.assertEqual(loaded, ["cuda", "mps"])
        self.assertEqual(batches[0][0], "mps")

    def test_accelerator_failures_fall_back_to_cpu(self):
        runner, loaded, _ = self._runner(
            FakeTorch(cuda=True, mps=True), {"cuda", "mps"},
        )
        batches = list(runner.iter_batches(["one"]))
        self.assertEqual(loaded, ["cuda", "mps", "cpu"])
        self.assertEqual(batches[0][0], "cpu")

    def test_mps_is_used_on_apple_silicon_when_cuda_is_absent(self):
        runner, loaded, _ = self._runner(FakeTorch(cuda=False, mps=True))
        self.assertEqual(list(runner.iter_batches(["one"]))[0][0], "mps")
        self.assertEqual(loaded, ["mps"])

    def test_explicit_unavailable_device_is_strict(self):
        runner, _, _ = self._runner(FakeTorch(cuda=False, mps=True), requested="cuda")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            list(runner.iter_batches(["one"]))

    def test_out_of_memory_reduces_batch_size(self):
        calls = []
        torch = FakeTorch(cuda=True, mps=False)

        class OOMModel:
            def encode(self, texts, **kwargs):
                calls.append(len(texts))
                if len(texts) > 1:
                    raise torch.OutOfMemoryError("out of memory")
                return [[1.0, 0.0] for _ in texts]

        runner = LocalSemanticEmbedder(
            "test-model", "auto", torch_module=torch,
            model_loader=lambda _name, device: OOMModel(), batch_size=8,
        )
        batches = list(runner.iter_batches(["one", "two", "three"]))
        self.assertEqual(sum(len(vectors) for _, vectors in batches), 3)
        self.assertIn(3, calls)
        self.assertGreaterEqual(calls.count(1), 4)  # probe plus three retried rows


if __name__ == "__main__":
    unittest.main()
