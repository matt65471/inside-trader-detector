"""Automated resolved-cohort collection, enrichment, labeling, and optional live mode."""
from __future__ import annotations

import hashlib
import json
import signal
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .api import APIError, PolymarketAPI
from .core import FEATURE_VERSION
from .db import Database
from .jobs import DeferredJob, Job, JobQueue
from .labeling import HORIZONS, HistoricalLabeler, ResolutionStore
from .sampled_backfill import BatchEmbedder, SampledBackfill, SampledBackfillConfig
from .service import BOOK_NOTIONAL_MICROUSD, Collector


class AutomatedPipeline:
    def __init__(
        self,
        db_path: str | Path,
        config: SampledBackfillConfig,
        *,
        market_workers: int = 2,
        enrichment_workers: int = 2,
        resolution_workers: int = 2,
        label_workers: int = 2,
        live: bool = False,
        live_interval: int = 60,
        clock: Callable[[], float] = time.time,
        api_factory: Callable[[], PolymarketAPI] = PolymarketAPI,
        embedder: BatchEmbedder | None = None,
        downstream: bool = True,
    ):
        self.db_path = Path(db_path)
        self.config = config
        self.market_workers = market_workers
        self.enrichment_workers = enrichment_workers
        self.resolution_workers = resolution_workers
        self.label_workers = label_workers
        self.live = live
        self.live_interval = live_interval
        self.clock = clock
        self.api_factory = api_factory
        self.embedder = embedder
        self.downstream = downstream
        self.stopping = threading.Event()

    def _db(self) -> Database:
        db = Database(self.db_path)
        db.initialize()
        return db

    @staticmethod
    def _job_remaining(db: Database, cohort: str, kinds: Iterable[str]) -> int:
        kinds = tuple(kinds)
        placeholders = ",".join("?" for _ in kinds)
        return int(db.row(
            f"""SELECT COUNT(*) n FROM jobs WHERE cohort_name=?
                AND job_kind IN ({placeholders}) AND status IN ('pending','leased')""",
            (cohort, *kinds),
        )["n"])

    @staticmethod
    def _dead(db: Database, cohort: str, kinds: Iterable[str]) -> int:
        kinds = tuple(kinds)
        placeholders = ",".join("?" for _ in kinds)
        return int(db.row(
            f"""SELECT COUNT(*) n FROM jobs WHERE cohort_name=?
                AND job_kind IN ({placeholders}) AND status='dead'""",
            (cohort, *kinds),
        )["n"])

    def _worker(
        self,
        kinds: tuple[str, ...],
        handler: Callable[[Database, PolymarketAPI, JobQueue, Job, str], None],
        done: threading.Event,
        name: str,
        claim_cohort: str,
    ) -> None:
        db = self._db()
        api = self.api_factory()
        queue = JobQueue(db, clock=self.clock)
        worker_id = f"{name}-{uuid.uuid4().hex[:8]}"
        try:
            while not self.stopping.is_set() and not done.is_set():
                job = queue.claim(
                    worker_id, kinds, cohort=claim_cohort, lease_seconds=900,
                )
                if job is None:
                    time.sleep(0.05)
                    continue
                try:
                    handler(db, api, queue, job, worker_id)
                except DeferredJob as exc:
                    queue.defer(job, worker_id, exc.available_at)
                except APIError as exc:
                    queue.fail(job, worker_id, exc, retryable=exc.retryable)
                except Exception as exc:
                    queue.fail(job, worker_id, exc, retryable=True)
                else:
                    queue.succeed(job, worker_id)
        finally:
            db.close()

    def _run_group(
        self,
        db: Database,
        kinds: tuple[str, ...],
        count: int,
        handler: Callable[[Database, PolymarketAPI, JobQueue, Job, str], None],
    ) -> int:
        done = threading.Event()
        threads = [
            threading.Thread(
                target=self._worker, args=(kinds, handler, done, f"{'-'.join(kinds)}-{index}"),
                kwargs={"claim_cohort": self.config.cohort},
                daemon=True,
            )
            for index in range(count)
        ]
        for thread in threads:
            thread.start()
        try:
            while not self.stopping.is_set():
                if self._job_remaining(db, self.config.cohort, kinds) == 0:
                    break
                time.sleep(0.05)
        finally:
            done.set()
            for thread in threads:
                thread.join(timeout=5)
        return self._dead(db, self.config.cohort, kinds)

    def _enqueue_resolution_verification(
        self, db: Database, conditions: Iterable[str] | None = None,
    ) -> None:
        queue = JobQueue(db, clock=self.clock)
        if conditions is None:
            rows = db.rows(
                """SELECT condition_id FROM backfill_cohort_markets cm
                   WHERE cohort_name=? AND eligible=1
                     AND NOT EXISTS (SELECT 1 FROM market_resolutions mr
                                     WHERE mr.condition_id=cm.condition_id AND mr.terminal=1)
                   ORDER BY condition_id""",
                (self.config.cohort,),
            )
            pending = [row["condition_id"] for row in rows]
        else:
            pending = [
                condition for condition in conditions
                if not (db.row(
                    "SELECT terminal FROM market_resolutions WHERE condition_id=?",
                    (condition,),
                ) or {"terminal": 0})["terminal"]
            ]
        for offset in range(0, len(pending), 20):
            batch = pending[offset:offset + 20]
            digest = hashlib.sha256("\0".join(batch).encode("utf-8")).hexdigest()[:20]
            queue.enqueue(
                "verify_resolution",
                f"verify_resolution:{self.config.cohort}:{digest}",
                {"conditions": batch}, cohort=self.config.cohort,
                entity_id=",".join(batch), priority=20,
            )

    def _verify_handler(
        self, db: Database, api: PolymarketAPI, _queue: JobQueue, job: Job, _worker: str,
    ) -> None:
        ResolutionStore(db, api, clock=self.clock).verify(
            self.config.cohort, list(job.payload["conditions"]),
        )

    def _enqueue_scans(self, db: Database) -> None:
        cohort = db.row("SELECT * FROM backfill_cohorts WHERE cohort_name=?", (self.config.cohort,))
        queue = JobQueue(db, clock=self.clock)
        for row in db.rows(
            """SELECT condition_id FROM backfill_cohort_markets
               WHERE cohort_name=? AND selection_status='selected'
                 AND fetch_status IN ('pending','failed')""",
            (self.config.cohort,),
        ):
            self._enqueue_scan(db, row["condition_id"], cohort=cohort, queue=queue)

    def _enqueue_scan(
        self, db: Database, condition: str, *,
        cohort: Mapping[str, Any] | None = None, queue: JobQueue | None = None,
    ) -> None:
        cohort = cohort or db.row(
            "SELECT * FROM backfill_cohorts WHERE cohort_name=?", (self.config.cohort,),
        )
        queue = queue or JobQueue(db, clock=self.clock)
        queue.enqueue(
            "scan_market", f"scan_market:{self.config.cohort}:{condition}",
            {
                "condition_id": condition,
                "window_start": cohort["window_start"],
                "window_end": cohort["window_end"],
            },
            cohort=self.config.cohort, entity_id=condition, priority=10,
        )

    def _needs_enrichment(self, db: Database, trade_key: str) -> bool:
        row = db.row(
            """SELECT t.enrichment_status,
                      EXISTS(SELECT 1 FROM trade_features f
                             WHERE f.trade_key=t.trade_key AND f.feature_version=?) has_features
               FROM trades t WHERE t.trade_key=?""",
            (FEATURE_VERSION, trade_key),
        )
        return bool(row and row["enrichment_status"] != "not_selected" and (
            row["enrichment_status"] in ("pending", "failed") or not row["has_features"]
        ))

    def _atomic_trade_jobs(
        self, cohort: str, *, live: bool,
    ) -> Callable[[sqlite3.Connection, Mapping[str, Any], bool], None]:
        """Build downstream jobs in the same transaction as a new trade."""
        def enqueue(
            connection: sqlite3.Connection, trade: Mapping[str, Any], selected: bool,
        ) -> None:
            now = int(self.clock())
            key = str(trade["trade_key"])
            job_cohort = f"{cohort}:live" if live else cohort
            if selected:
                JobQueue.enqueue_in_transaction(
                    connection, "enrich_trade", f"enrich_trade:{key}:v{FEATURE_VERSION}",
                    {"trade_key": key}, now=now, cohort=job_cohort, entity_id=key,
                )
            for horizon in HORIZONS:
                JobQueue.enqueue_in_transaction(
                    connection, "observe_entry", f"observe_entry:{key}:{horizon}",
                    {"trade_key": key, "horizon": horizon, "live": live},
                    now=now, cohort=job_cohort, entity_id=key, priority=30 if live else 5,
                    available_at=int(trade["trade_ts"]) + horizon if live else now,
                )
            if live and int(trade["notional_microusd"]) >= BOOK_NOTIONAL_MICROUSD:
                JobQueue.enqueue_in_transaction(
                    connection, "capture_book", f"capture_book:{key}:post_detection",
                    {"trade_key": key}, now=now, cohort=job_cohort, entity_id=key, priority=40,
                )
            if live:
                market = connection.execute(
                    "SELECT end_ts FROM markets WHERE condition_id=?",
                    (trade["condition_id"],),
                ).fetchone()
                due = max(now, int(market["end_ts"])) if market and market["end_ts"] else now
                JobQueue.enqueue_in_transaction(
                    connection, "resolve_market", f"resolve_market:{trade['condition_id']}",
                    {"condition_id": trade["condition_id"]}, now=now, cohort=job_cohort,
                    entity_id=str(trade["condition_id"]), available_at=due,
                )
        return enqueue

    def _enqueue_trade_jobs(
        self, db: Database, cohort: str, *, condition: str | None = None, live: bool = False,
    ) -> None:
        queue = JobQueue(db, clock=self.clock)
        job_cohort = f"{cohort}:live" if live else cohort
        if live:
            sql = "SELECT * FROM trades WHERE source_mode='live'"
            params: tuple[Any, ...] = ()
        else:
            sql = """SELECT t.* FROM backfill_cohort_markets cm
                     JOIN backfill_cohorts c USING(cohort_name)
                     JOIN trades t ON t.condition_id=cm.condition_id
                                  AND t.trade_ts BETWEEN c.window_start AND c.window_end
                     WHERE cm.cohort_name=? AND cm.selection_status='selected'"""
            params = (cohort,)
            if condition:
                sql += " AND t.condition_id=?"
                params += (condition,)
        for trade in db.rows(sql, params):
            key = trade["trade_key"]
            if self._needs_enrichment(db, key):
                queue.enqueue(
                    "enrich_trade", f"enrich_trade:{key}:v{FEATURE_VERSION}",
                    {"trade_key": key}, cohort=job_cohort, entity_id=key,
                    reopen_succeeded=True,
                )
            for horizon in HORIZONS:
                label = db.row(
                    "SELECT status FROM labels WHERE trade_key=? AND horizon_seconds=?",
                    (key, horizon),
                )
                if not label or label["status"] in ("pending", "observed_pending_resolution"):
                    queue.enqueue(
                        "observe_entry", f"observe_entry:{key}:{horizon}",
                        {"trade_key": key, "horizon": horizon, "live": live},
                        cohort=job_cohort, entity_id=key, priority=30 if live else 5,
                        available_at=int(trade["trade_ts"]) + horizon if live else int(self.clock()),
                        reopen_succeeded=True,
                    )
            if live and int(trade["notional_microusd"]) >= BOOK_NOTIONAL_MICROUSD:
                queue.enqueue(
                    "capture_book", f"capture_book:{key}:post_detection",
                    {"trade_key": key}, cohort=job_cohort, entity_id=key, priority=40,
                )
            if live:
                market = db.row("SELECT end_ts FROM markets WHERE condition_id=?", (trade["condition_id"],))
                due = max(int(self.clock()), int(market["end_ts"])) if market and market["end_ts"] else int(self.clock())
                queue.enqueue(
                    "resolve_market", f"resolve_market:{trade['condition_id']}",
                    {"condition_id": trade["condition_id"]}, cohort=job_cohort,
                    entity_id=trade["condition_id"], available_at=due,
                )

    def _scan_handler(
        self, db: Database, api: PolymarketAPI, queue: JobQueue, job: Job, worker: str,
    ) -> None:
        condition = job.payload["condition_id"]
        row = db.row(
            "SELECT * FROM backfill_cohort_markets WHERE cohort_name=? AND condition_id=?",
            (self.config.cohort, condition),
        )
        if not row:
            raise ValueError(f"Unknown cohort market: {condition}")
        checkpoint = f"sampled:{self.config.cohort}:{condition}"
        saved = db.checkpoint(checkpoint)
        collector = Collector(
            db, api, clock=self.clock, capture_books_inline=False,
            on_trade_stored=(
                self._atomic_trade_jobs(self.config.cohort, live=False)
                if self.downstream else None
            ),
        )
        counts = collector.collect_window(
            start=int(job.payload["window_start"]), end=int(job.payload["window_end"]),
            source_mode="backfill", condition=condition, checkpoint_name=checkpoint,
            initial_cursor=saved["cursor"] if saved else None,
            market_raw=json.loads(row["raw_json"]),
            progress=lambda: queue.renew(job, worker, lease_seconds=900),
        )
        queue.renew(job, worker, lease_seconds=900)
        if counts.errors:
            with db.connection:
                db.connection.execute(
                    """UPDATE backfill_cohort_markets SET fetch_status='failed',last_error=?
                       WHERE cohort_name=? AND condition_id=?""",
                    (f"Market scan failed for {condition}", self.config.cohort, condition),
                )
            raise APIError(f"Market scan failed for {condition}", retryable=True)
        count = db.row(
            "SELECT COUNT(*) n FROM trades WHERE condition_id=? AND trade_ts BETWEEN ? AND ?",
            (condition, job.payload["window_start"], job.payload["window_end"]),
        )["n"]
        with db.connection:
            db.connection.execute(
                """UPDATE backfill_cohort_markets SET fetch_status='complete',
                          qualifying_trade_count=?,last_error=NULL
                   WHERE cohort_name=? AND condition_id=?""",
                (count, self.config.cohort, condition),
            )
        if self.downstream:
            self._enqueue_trade_jobs(db, self.config.cohort, condition=condition)

    def _enrich_handler(
        self, db: Database, api: PolymarketAPI, _queue: JobQueue, job: Job, _worker: str,
    ) -> None:
        collector = Collector(db, api, clock=self.clock, capture_books_inline=False)
        if not collector.enrich_trade(job.payload["trade_key"]):
            raise APIError(f"Enrichment failed for {job.payload['trade_key']}", retryable=True)

    def _label_handler(
        self, db: Database, api: PolymarketAPI, _queue: JobQueue, job: Job, _worker: str,
    ) -> None:
        labeler = HistoricalLabeler(db, api, clock=self.clock)
        if job.job_kind == "observe_entry":
            if job.payload.get("live"):
                labeler.observe_live(job.payload["trade_key"], int(job.payload["horizon"]))
            else:
                labeler.observe(
                    self.config.cohort, job.payload["trade_key"], int(job.payload["horizon"]),
                )
            return
        if job.job_kind == "capture_book":
            if not Collector(db, api, clock=self.clock, capture_books_inline=False).capture_book(
                job.payload["trade_key"]
            ):
                raise APIError(f"Book capture failed for {job.payload['trade_key']}", retryable=True)
            return
        if job.job_kind == "resolve_market":
            condition = job.payload["condition_id"]
            ResolutionStore(db, api, clock=self.clock).verify("__live__", [condition])
            row = db.row("SELECT terminal FROM market_resolutions WHERE condition_id=?", (condition,))
            if not row or not row["terminal"]:
                raise DeferredJob(int(self.clock()) + 6 * 3600)
            labeler.finalize_market(condition)
            return
        raise ValueError(f"Unsupported label job: {job.job_kind}")

    def _historical_workers(self, db: Database) -> int:
        scan_done = threading.Event()
        all_done = threading.Event()
        specs = [
            (("scan_market",), self.market_workers, self._scan_handler, scan_done, "market"),
            (("enrich_trade",), self.enrichment_workers, self._enrich_handler, all_done, "enrich"),
            (("observe_entry",), self.label_workers, self._label_handler, all_done, "label"),
        ]
        threads: list[threading.Thread] = []
        for kinds, count, handler, done, prefix in specs:
            for index in range(count):
                thread = threading.Thread(
                    target=self._worker, args=(kinds, handler, done, f"{prefix}-{index}"), daemon=True,
                    kwargs={"claim_cohort": self.config.cohort},
                )
                threads.append(thread)
                thread.start()
        try:
            required = ("scan_market", "enrich_trade", "observe_entry")
            while not self.stopping.is_set() and self._job_remaining(db, self.config.cohort, required):
                time.sleep(0.05)
        finally:
            scan_done.set()
            all_done.set()
            for thread in threads:
                thread.join(timeout=5)
        return self._dead(db, self.config.cohort, ("scan_market", "enrich_trade", "observe_entry"))

    def _streaming_historical_workers(
        self, db: Database, sampled: SampledBackfill,
    ) -> int:
        """Run scan/downstream consumers while the main thread produces selections."""
        # Reconstruct durable work before consumers start so reconciliation cannot
        # race a worker that is committing the same job's successful result.
        self._enqueue_scans(db)
        if self.downstream:
            self._enqueue_trade_jobs(db, self.config.cohort)
        done = threading.Event()
        specs = [(("scan_market",), self.market_workers, self._scan_handler, "market")]
        if self.downstream:
            specs.extend([
                (("enrich_trade",), self.enrichment_workers, self._enrich_handler, "enrich"),
                (("observe_entry",), self.label_workers, self._label_handler, "label"),
            ])
        threads: list[threading.Thread] = []
        for kinds, count, handler, prefix in specs:
            for index in range(count):
                thread = threading.Thread(
                    target=self._worker,
                    args=(kinds, handler, done, f"{prefix}-{index}"),
                    kwargs={"claim_cohort": self.config.cohort}, daemon=True,
                )
                threads.append(thread)
                thread.start()

        def prepare(conditions: list[str]) -> None:
            if not self.config.resolution_required or not conditions:
                return
            sampled._update_cohort(resolution_status="running")
            self._enqueue_resolution_verification(db, conditions)
            if self._run_group(
                db, ("verify_resolution",), self.resolution_workers, self._verify_handler,
            ):
                sampled._update_cohort(
                    resolution_status="failed", last_error="resolution jobs failed",
                )
                raise RuntimeError("One or more resolution jobs failed")

        try:
            sampled.stream_select(
                prepare_candidates=prepare if self.config.resolution_required else None,
                on_selected=lambda condition: self._enqueue_scan(db, condition),
            )
            required = ("scan_market", "enrich_trade", "observe_entry") if self.downstream else (
                "scan_market",
            )
            while not self.stopping.is_set() and self._job_remaining(
                db, self.config.cohort, required,
            ):
                time.sleep(0.05)
            return self._dead(db, self.config.cohort, required)
        finally:
            done.set()
            for thread in threads:
                thread.join(timeout=5)

    def _poll_live_handler(
        self, db: Database, api: PolymarketAPI, _queue: JobQueue, job: Job, _worker: str,
    ) -> None:
        now = int(self.clock())
        checkpoint = db.checkpoint("live")
        last = int(checkpoint["last_trade_ts"]) if checkpoint and checkpoint["last_trade_ts"] else None
        start = max(1, last - 60 if last else now - 120)
        collector = Collector(
            db, api, clock=self.clock, capture_books_inline=False,
            on_trade_stored=self._atomic_trade_jobs(self.config.cohort, live=True),
        )
        counts = collector.collect_window(
            start=start, end=now, source_mode="live", checkpoint_name="live",
            initial_cursor=checkpoint["cursor"] if checkpoint else None,
        )
        if counts.errors:
            raise APIError("Live poll failed", retryable=True)
        self._enqueue_trade_jobs(db, self.config.cohort, live=True)
        raise DeferredJob(now + self.live_interval)

    def _run_live(self, db: Database) -> int:
        queue = JobQueue(db, clock=self.clock)
        queue.enqueue(
            "poll_live", f"poll_live:{self.config.cohort}", {}, cohort=self.config.cohort,
            priority=50,
        )
        done = threading.Event()
        threads: list[threading.Thread] = []
        live_cohort = f"{self.config.cohort}:live"
        JobQueue(db, clock=self.clock).recover_expired(live_cohort)
        specs = [
            (("poll_live",), 1, self._poll_live_handler, "poll-live", self.config.cohort),
            (("enrich_trade",), self.enrichment_workers, self._enrich_handler, "live-enrich", live_cohort),
            (("observe_entry", "capture_book", "resolve_market"), self.label_workers, self._label_handler, "live-label", live_cohort),
        ]
        for kinds, count, handler, prefix, claim_cohort in specs:
            for index in range(count):
                thread = threading.Thread(
                    target=self._worker, args=(kinds, handler, done, f"{prefix}-{index}"), daemon=True,
                    kwargs={"claim_cohort": claim_cohort},
                )
                threads.append(thread)
                thread.start()
        try:
            while not self.stopping.is_set():
                time.sleep(0.2)
        finally:
            done.set()
            for thread in threads:
                thread.join(timeout=5)
        return 130

    def run(self) -> int:
        db = self._db()
        previous = signal.signal(signal.SIGINT, lambda *_: self.stopping.set())
        try:
            JobQueue(db, clock=self.clock).recover_expired(self.config.cohort)
            api = self.api_factory()
            collector = Collector(db, api, clock=self.clock, capture_books_inline=False)
            sampled = SampledBackfill(
                db, collector, self.config, clock=self.clock, embedder=self.embedder,
            )
            cohort = sampled._cohort()
            if cohort["history_mode"] == "lifetime":
                if self._streaming_historical_workers(db, sampled):
                    sampled._update_cohort(
                        phase="fetching", last_error="one or more pipeline jobs are dead",
                    )
                    return 1
                sampled._update_cohort(phase="complete", last_error=None)
                label = "Automated historical cohort" if self.downstream else "Sampled backfill cohort"
                print(f"{label} {self.config.cohort} is complete", flush=True)
                if self.live and self.downstream and not self.stopping.is_set():
                    return self._run_live(db)
                return 130 if self.stopping.is_set() else 0
            sampled.discover(cohort)
            cohort = sampled._cohort()
            if self.config.resolution_required and cohort["resolution_status"] != "complete":
                sampled._update_cohort(resolution_status="running")
                self._enqueue_resolution_verification(db)
                if self._run_group(
                    db, ("verify_resolution",), self.resolution_workers, self._verify_handler,
                ):
                    sampled._update_cohort(resolution_status="failed", last_error="resolution jobs failed")
                    return 1
                sampled._update_cohort(resolution_status="complete", last_error=None)
            cohort = sampled._cohort()
            if cohort["phase"] == "embedding":
                sampled.ensure_embeddings()
            sampled.select()
            self._enqueue_scans(db)
            if self.downstream:
                self._enqueue_trade_jobs(db, self.config.cohort)
            dead = (
                self._historical_workers(db)
                if self.downstream else
                self._run_group(db, ("scan_market",), self.market_workers, self._scan_handler)
            )
            if dead:
                sampled._update_cohort(phase="fetching", last_error="one or more pipeline jobs are dead")
                return 1
            sampled._update_cohort(phase="complete", last_error=None)
            print(f"Automated historical cohort {self.config.cohort} is complete", flush=True)
            if self.live and not self.stopping.is_set():
                return self._run_live(db)
            return 130 if self.stopping.is_set() else 0
        except KeyboardInterrupt:
            self.stopping.set()
            return 130
        except (APIError, RuntimeError, ValueError) as exc:
            try:
                with db.connection:
                    db.connection.execute(
                        "UPDATE backfill_cohorts SET last_error=?,updated_at=? WHERE cohort_name=?",
                        (str(exc), int(self.clock()), self.config.cohort),
                    )
            except sqlite3.Error:
                pass
            print(f"Automated pipeline stopped: {exc}", file=sys.stderr)
            return 1
        finally:
            signal.signal(signal.SIGINT, previous)
            db.close()
