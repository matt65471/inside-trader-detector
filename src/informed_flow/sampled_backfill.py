"""Resumable, category-stratified historical market sampling."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from .api import APIError
from .core import canonical_json, classify_five_minute_updown, first, parse_timestamp
from .db import Database
from .service import Collector


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_CATEGORIES = (
    "sports", "crypto", "weather", "culture", "finance", "geopolitics", "politics",
)
CATEGORY_ALIASES = {
    "elections": "politics",
    "economy": "finance",
    "business": "finance",
    "world": "geopolitics",
}


@dataclass(frozen=True)
class SampledBackfillConfig:
    cohort: str
    days: int = 90
    markets_per_category: int = 1000
    seed: int = 0
    similarity_threshold: float = 0.90
    embedding_device: str = "auto"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    resolution_required: bool = False


class BatchEmbedder(Protocol):
    failures: list[str]

    def iter_batches(self, texts: Sequence[str]) -> Iterator[tuple[str, list[list[float]]]]:
        ...


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _random_rank(seed: int, category: str, condition_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{condition_id}".encode("utf-8")).hexdigest()


def _event(market: Mapping[str, Any]) -> Mapping[str, Any]:
    events = first(market, "events", default=[]) or []
    return events[0] if events and isinstance(events[0], Mapping) else {}


def raw_market_category(market: Mapping[str, Any]) -> str | None:
    event = _event(market)
    direct = first(event, "category", default=first(market, "category"))
    if direct:
        return str(direct).strip().lower()
    known = set(TARGET_CATEGORIES) | set(CATEGORY_ALIASES) | {"tech", "science"}
    for tag in first(market, "tags", default=[]) or []:
        if not isinstance(tag, Mapping):
            continue
        for key in ("slug", "label"):
            value = str(first(tag, key, default="")).strip().lower()
            if value in known:
                return value
    return None


def canonical_market_category(market: Mapping[str, Any]) -> tuple[str | None, str | None]:
    raw = raw_market_category(market)
    if first(market, "sportsMarketType", "sports_market_type", "gameStartTime"):
        return raw, "sports"
    canonical = CATEGORY_ALIASES.get(raw or "", raw)
    return raw, canonical if canonical in TARGET_CATEGORIES else None


def _normalized_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not vector or not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Embedding vector is empty or cannot be normalized")
    return [value / magnitude for value in vector]


def _pack_vector(values: Iterable[float]) -> tuple[int, bytes]:
    packed = array("f", _normalized_vector(values))
    if sys.byteorder != "little":
        packed.byteswap()
    return len(packed), packed.tobytes()


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != dimensions:
        raise ValueError("Stored embedding dimensions do not match its vector")
    return list(values)


class LocalSemanticEmbedder:
    """Sentence Transformers runner with explicit accelerator fallback."""

    def __init__(
        self,
        model_name: str,
        requested_device: str = "auto",
        *,
        batch_size: int = 64,
        torch_module: Any | None = None,
        model_loader: Callable[..., Any] | None = None,
    ):
        self.model_name = model_name
        self.requested_device = requested_device
        self.batch_size = batch_size
        self.failures: list[str] = []
        try:
            if torch_module is None:
                import torch as torch_module  # type: ignore[no-redef]
            if model_loader is None:
                from sentence_transformers import SentenceTransformer
                model_loader = SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'Semantic sampling requires the optional dependency: pip install -e ".[semantic]"'
            ) from exc
        self.torch = torch_module
        self.model_loader = model_loader

    def _available(self, device: str) -> bool:
        if device == "cuda":
            return bool(self.torch.cuda.is_available())
        if device == "mps":
            backend = getattr(getattr(self.torch, "backends", None), "mps", None)
            return bool(backend and backend.is_available())
        return device == "cpu"

    def _devices(self) -> list[str]:
        if self.requested_device != "auto":
            if not self._available(self.requested_device):
                raise RuntimeError(f"Requested embedding device is unavailable: {self.requested_device}")
            return [self.requested_device]
        return [device for device in ("cuda", "mps", "cpu") if self._available(device)]

    def _is_oom(self, exc: BaseException) -> bool:
        oom_type = getattr(self.torch, "OutOfMemoryError", None)
        return bool((oom_type and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower())

    @staticmethod
    def _rows(value: Any) -> list[list[float]]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [[float(item) for item in row] for row in value]

    def iter_batches(self, texts: Sequence[str]) -> Iterator[tuple[str, list[list[float]]]]:
        position = 0
        devices = self._devices()
        last_error: BaseException | None = None
        for device in devices:
            batch_size = self.batch_size
            try:
                model = self.model_loader(self.model_name, device=device)
                model.encode(
                    ["embedding device probe"], batch_size=1, normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                last_error = exc
                self.failures.append(f"{device}: initialization/probe failed: {exc}")
                if self.requested_device != "auto":
                    raise RuntimeError(self.failures[-1]) from exc
                continue
            while position < len(texts):
                size = min(batch_size, len(texts) - position)
                try:
                    vectors = model.encode(
                        list(texts[position:position + size]), batch_size=size,
                        normalize_embeddings=True, convert_to_numpy=True,
                    )
                    rows = self._rows(vectors)
                    if len(rows) != size:
                        raise RuntimeError("Embedding model returned the wrong number of vectors")
                    yield device, rows
                    position += size
                except Exception as exc:
                    last_error = exc
                    if self._is_oom(exc) and size > 1:
                        batch_size = max(1, size // 2)
                        self.failures.append(
                            f"{device}: out of memory at batch {size}; retrying with {batch_size}"
                        )
                        continue
                    self.failures.append(f"{device}: inference failed: {exc}")
                    if self.requested_device != "auto":
                        raise RuntimeError(self.failures[-1]) from exc
                    break
            if position == len(texts):
                return
        raise RuntimeError(
            "All embedding devices failed" + (f": {last_error}" if last_error else "")
        )


class SampledBackfill:
    def __init__(
        self,
        db: Database,
        collector: Collector,
        config: SampledBackfillConfig,
        *,
        clock: Callable[[], float] = time.time,
        embedder: BatchEmbedder | None = None,
    ):
        self.db = db
        self.collector = collector
        self.config = config
        self.clock = clock
        self.embedder = embedder

    @property
    def threshold_ppm(self) -> int:
        return round(self.config.similarity_threshold * 1_000_000)

    def _cohort(self) -> Mapping[str, Any]:
        row = self.db.row("SELECT * FROM backfill_cohorts WHERE cohort_name=?", (self.config.cohort,))
        if row:
            expected = {
                "days": self.config.days,
                "category_limit": self.config.markets_per_category,
                "seed": self.config.seed,
                "embedding_model": self.config.embedding_model,
                "similarity_threshold_ppm": self.threshold_ppm,
                "requested_device": self.config.embedding_device,
                "resolution_required": int(self.config.resolution_required),
            }
            changed = [key for key, value in expected.items() if row[key] != value]
            if changed:
                raise ValueError(
                    f"Cohort {self.config.cohort!r} already exists with different "
                    f"configuration: {', '.join(changed)}"
                )
            return row
        legacy = self.db.checkpoint(f"market_backfill_{self.config.days}d")
        end = int(legacy["window_end"]) if legacy and legacy["window_end"] else int(self.clock())
        start = int(legacy["window_start"]) if legacy and legacy["window_start"] else end - self.config.days * 86400
        now = int(self.clock())
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO backfill_cohorts(
                       cohort_name,window_start,window_end,days,category_limit,seed,
                       embedding_model,similarity_threshold_ppm,requested_device,
                       resolution_required,resolution_status,phase,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'discovering',?,?)""",
                (
                    self.config.cohort, start, end, self.config.days,
                    self.config.markets_per_category, self.config.seed,
                    self.config.embedding_model, self.threshold_ppm,
                    self.config.embedding_device, int(self.config.resolution_required),
                    "pending" if self.config.resolution_required else "not_required", now, now,
                ),
            )
        return self.db.row("SELECT * FROM backfill_cohorts WHERE cohort_name=?", (self.config.cohort,))

    def _update_cohort(self, **values: Any) -> None:
        values["updated_at"] = int(self.clock())
        assignments = ",".join(f"{key}=?" for key in values)
        with self.db.connection:
            self.db.connection.execute(
                f"UPDATE backfill_cohorts SET {assignments} WHERE cohort_name=?",
                (*values.values(), self.config.cohort),
            )

    def _discover_market(self, raw: Mapping[str, Any], start: int, end: int) -> None:
        condition = first(raw, "conditionId", "condition_id")
        if not condition or not str(condition).strip():
            self.db.record_error(
                None, "sampled_discovery",
                canonical_json({
                    "cohort": self.config.cohort,
                    "reason": "missing_condition_id",
                    "raw_market": raw,
                }),
                endpoint="/markets", entity_type="market",
                entity_id=str(raw.get("id")) if raw.get("id") is not None else None,
            )
            print(
                f"Skipping Gamma market id={raw.get('id')}: missing condition ID "
                "(raw listing retained in collection_errors)",
                flush=True,
            )
            return
        condition = str(condition).strip().lower()
        event = _event(raw)
        title = str(first(raw, "question", "title", default="")).strip()
        created = parse_timestamp(first(raw, "createdAt", "created_at"))
        market_end = parse_timestamp(first(raw, "endDate", "end_date", "endDateIso"))
        raw_category, category = canonical_market_category(raw)
        excluded, evidence = classify_five_minute_updown(raw, raw)
        eligible = True
        reason = None
        status = "candidate"
        if excluded:
            eligible, reason, status = False, evidence or "five_minute_updown", "excluded"
        elif category is None:
            eligible, reason, status = False, "category_not_targeted", "not_target"
        elif not title:
            eligible, reason, status = False, "missing_title", "ineligible"
        elif created is not None and created > end:
            eligible, reason, status = False, "created_after_window", "ineligible"
        elif market_end is not None and market_end < start:
            eligible, reason, status = False, "ended_before_window", "ineligible"
        normalized = normalize_title(title)
        now = int(self.clock())
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO backfill_cohort_markets(
                       cohort_name,condition_id,event_id,event_slug,title,raw_category,
                       canonical_category,created_ts,end_ts,raw_json,eligible,
                       eligibility_reason,random_rank,embedding_text_hash,selection_status,
                       fetch_status,discovered_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'not_selected',?)
                   ON CONFLICT(cohort_name,condition_id) DO NOTHING""",
                (
                    self.config.cohort, condition,
                    str(first(event, "id", default="")) or None,
                    str(first(event, "slug", default="")) or None,
                    title, raw_category, category, created, market_end, canonical_json(raw),
                    int(eligible), reason,
                    _random_rank(self.config.seed, category or "", condition) if eligible else None,
                    _text_hash(normalized) if eligible else None,
                    status, now,
                ),
            )

    def discover(self, cohort: Mapping[str, Any]) -> None:
        if cohort["discovery_complete"]:
            return
        cursor = cohort["discovery_cursor"]
        seen: set[str | None] = set()
        start, end = int(cohort["window_start"]), int(cohort["window_end"])
        while True:
            if cursor in seen:
                raise APIError("Gamma repeated a sampled discovery cursor")
            seen.add(cursor)
            print(f"Sampled discovery page cursor={cursor or 'start'}", flush=True)
            page = self.collector.api.markets_page(cursor, closed=True)
            for raw in page.rows:
                self._discover_market(raw, start, end)
            cursor = page.next_cursor
            if cursor:
                self._update_cohort(discovery_cursor=cursor, last_error=None)
                continue
            self._update_cohort(
                discovery_cursor=None, discovery_complete=1, phase="embedding", last_error=None,
            )
            return

    def ensure_embeddings(self) -> None:
        rows = self.db.rows(
            """SELECT cm.embedding_text_hash,MIN(cm.title) title
               FROM backfill_cohort_markets cm
               LEFT JOIN semantic_embeddings se
                 ON se.model_name=? AND se.text_hash=cm.embedding_text_hash
               WHERE cm.cohort_name=? AND cm.eligible=1 AND se.text_hash IS NULL
               GROUP BY cm.embedding_text_hash
               ORDER BY cm.embedding_text_hash""",
            (self.config.embedding_model, self.config.cohort),
        )
        if not rows:
            cached_devices = self.db.rows(
                """SELECT DISTINCT se.generation_device
                   FROM backfill_cohort_markets cm JOIN semantic_embeddings se
                     ON se.model_name=? AND se.text_hash=cm.embedding_text_hash
                   WHERE cm.cohort_name=? AND cm.eligible=1
                   ORDER BY se.generation_device""",
                (self.config.embedding_model, self.config.cohort),
            )
            self._update_cohort(phase="selecting", last_error=None)
            if cached_devices:
                self._update_cohort(
                    effective_device=",".join(row["generation_device"] for row in cached_devices)
                )
            return
        embedder = self.embedder or LocalSemanticEmbedder(
            self.config.embedding_model, self.config.embedding_device,
        )
        texts = [normalize_title(row["title"]) for row in rows]
        position = 0
        devices: list[str] = []
        try:
            for device, vectors in embedder.iter_batches(texts):
                if device not in devices:
                    devices.append(device)
                if position + len(vectors) > len(rows):
                    raise RuntimeError("Embedding runner returned too many vectors")
                with self.db.connection:
                    for row, vector in zip(rows[position:position + len(vectors)], vectors):
                        dimensions, blob = _pack_vector(vector)
                        self.db.connection.execute(
                            """INSERT OR IGNORE INTO semantic_embeddings(
                                   model_name,text_hash,dimensions,vector_blob,generation_device,created_at
                               ) VALUES (?,?,?,?,?,?)""",
                            (
                                self.config.embedding_model, row["embedding_text_hash"], dimensions,
                                blob, device, int(self.clock()),
                            ),
                        )
                position += len(vectors)
                self._update_cohort(effective_device=",".join(devices))
                print(f"Embedded titles: {position}/{len(rows)} on {device}", flush=True)
        except Exception:
            self._update_cohort(
                embedding_device_log=canonical_json(getattr(embedder, "failures", []))
            )
            raise
        if position != len(rows):
            raise RuntimeError("Embedding runner ended before all titles were embedded")
        failures = getattr(embedder, "failures", [])
        cached_devices = self.db.rows(
            """SELECT DISTINCT se.generation_device
               FROM backfill_cohort_markets cm JOIN semantic_embeddings se
                 ON se.model_name=? AND se.text_hash=cm.embedding_text_hash
               WHERE cm.cohort_name=? AND cm.eligible=1 ORDER BY se.generation_device""",
            (self.config.embedding_model, self.config.cohort),
        )
        self._update_cohort(
            phase="selecting",
            effective_device=",".join(row["generation_device"] for row in cached_devices),
            embedding_device_log=canonical_json(failures), last_error=None,
        )

    @staticmethod
    def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def select(self) -> None:
        cohort = self._cohort()
        if cohort["phase"] in ("fetching", "complete"):
            return
        decisions: list[tuple[str, str, str | None, str | None, int | None]] = []
        try:
            import numpy as np
        except ImportError:  # The semantic extra supplies NumPy; tiny fake test sets can use Python.
            np = None
        for category in TARGET_CATEGORIES:
            rows = self.db.rows(
                """SELECT cm.*,se.dimensions,se.vector_blob
                   FROM backfill_cohort_markets cm
                   JOIN semantic_embeddings se
                     ON se.model_name=? AND se.text_hash=cm.embedding_text_hash
                   WHERE cm.cohort_name=? AND cm.eligible=1 AND cm.canonical_category=?
                   ORDER BY cm.random_rank""",
                (self.config.embedding_model, self.config.cohort, category),
            )
            accepted_ids: list[str] = []
            accepted_vectors: list[list[float]] = []
            accepted_matrix = None
            event_ids: dict[str, str] = {}
            event_slugs: dict[str, str] = {}
            for row in rows:
                condition = row["condition_id"]
                if len(accepted_ids) >= self.config.markets_per_category:
                    decisions.append((condition, "not_selected", "category_limit", None, None))
                    continue
                duplicate = None
                selection_reason = None
                similarity_ppm = None
                if row["event_id"] and row["event_id"] in event_ids:
                    duplicate = event_ids[row["event_id"]]
                    selection_reason = "same_event_id"
                    similarity_ppm = 1_000_000
                elif row["event_slug"] and row["event_slug"] in event_slugs:
                    duplicate = event_slugs[row["event_slug"]]
                    selection_reason = "same_event_slug"
                    similarity_ppm = 1_000_000
                vector = _unpack_vector(row["vector_blob"], row["dimensions"])
                if duplicate is None and accepted_ids:
                    if np is not None:
                        assert accepted_matrix is not None
                        similarities = accepted_matrix[:len(accepted_ids)] @ np.asarray(vector, dtype=np.float32)
                        match = int(similarities.argmax())
                        similarity, prior_id = float(similarities[match]), accepted_ids[match]
                    else:
                        scored = [
                            (self._similarity(vector, prior), prior_id)
                            for prior_id, prior in zip(accepted_ids, accepted_vectors)
                        ]
                        similarity, prior_id = max(scored)
                    similarity_ppm = max(-1_000_000, min(1_000_000, round(similarity * 1_000_000)))
                    if similarity_ppm >= self.threshold_ppm:
                        duplicate = prior_id
                        selection_reason = "semantic_similarity"
                if duplicate is not None:
                    decisions.append(
                        (condition, "redundant", selection_reason, duplicate, similarity_ppm)
                    )
                    continue
                decisions.append((condition, "selected", None, None, None))
                if np is not None:
                    if accepted_matrix is None:
                        accepted_matrix = np.empty(
                            (self.config.markets_per_category, len(vector)), dtype=np.float32
                        )
                    accepted_matrix[len(accepted_ids)] = vector
                else:
                    accepted_vectors.append(vector)
                accepted_ids.append(condition)
                if row["event_id"]:
                    event_ids[row["event_id"]] = condition
                if row["event_slug"]:
                    event_slugs[row["event_slug"]] = condition
            print(f"Selected {len(accepted_ids)} nonredundant {category} markets", flush=True)
        with self.db.connection:
            self.db.connection.executemany(
                """UPDATE backfill_cohort_markets SET
                       selection_status=?,selection_reason=?,duplicate_of_condition_id=?,similarity_ppm=?,
                       fetch_status=CASE WHEN ?='selected' THEN 'pending' ELSE 'not_selected' END
                   WHERE cohort_name=? AND condition_id=?""",
                [
                    (status, reason, duplicate, similarity, status, self.config.cohort, condition)
                    for condition, status, reason, duplicate, similarity in decisions
                ],
            )
            self.db.connection.execute(
                """UPDATE backfill_cohorts SET phase='fetching',last_error=NULL,updated_at=?
                   WHERE cohort_name=?""",
                (int(self.clock()), self.config.cohort),
            )

    def fetch(self) -> int:
        cohort = self._cohort()
        start, end = int(cohort["window_start"]), int(cohort["window_end"])
        rows = self.db.rows(
            """SELECT * FROM backfill_cohort_markets
               WHERE cohort_name=? AND selection_status='selected'
                 AND fetch_status IN ('pending','failed')
               ORDER BY canonical_category,random_rank""",
            (self.config.cohort,),
        )
        for index, row in enumerate(rows, 1):
            condition = row["condition_id"]
            checkpoint_name = f"sampled:{self.config.cohort}:{condition}"
            saved = self.db.checkpoint(checkpoint_name)
            print(
                f"Fetching sampled market {index}/{len(rows)} category={row['canonical_category']} "
                f"condition={condition}", flush=True,
            )
            counts = self.collector.collect_window(
                start=start, end=end, source_mode="backfill", condition=condition,
                checkpoint_name=checkpoint_name,
                initial_cursor=saved["cursor"] if saved else None,
                market_raw=json.loads(row["raw_json"]),
            )
            count = self.db.row(
                "SELECT COUNT(*) n FROM trades WHERE condition_id=? AND trade_ts BETWEEN ? AND ?",
                (condition, start, end),
            )["n"]
            if counts.errors:
                error = self.db.row(
                    "SELECT error_message FROM collection_errors ORDER BY error_id DESC LIMIT 1"
                )
                message = error["error_message"] if error else "Trade fetch failed"
                with self.db.connection:
                    self.db.connection.execute(
                        """UPDATE backfill_cohort_markets SET fetch_status='failed',
                               qualifying_trade_count=?,last_error=?
                           WHERE cohort_name=? AND condition_id=?""",
                        (count, message, self.config.cohort, condition),
                    )
                self._update_cohort(phase="fetching", last_error=message)
                return 1
            with self.db.connection:
                self.db.connection.execute(
                    """UPDATE backfill_cohort_markets SET fetch_status='complete',
                           qualifying_trade_count=?,last_error=NULL
                       WHERE cohort_name=? AND condition_id=?""",
                    (count, self.config.cohort, condition),
                )
        self._update_cohort(phase="complete", last_error=None)
        print(f"Sampled backfill cohort {self.config.cohort} is complete", flush=True)
        return 0

    def run(self) -> int:
        try:
            cohort = self._cohort()
            if cohort["phase"] == "complete":
                print(f"Sampled backfill cohort {self.config.cohort} is already complete")
                return 0
            self.discover(cohort)
            cohort = self._cohort()
            if self.config.resolution_required and cohort["resolution_status"] != "complete":
                raise RuntimeError(
                    "This cohort requires resolution verification; use the automated run command"
                )
            if cohort["phase"] == "embedding":
                self.ensure_embeddings()
            self.select()
            return self.fetch()
        except KeyboardInterrupt:
            self._update_cohort(last_error="interrupted")
            print("Sampled backfill interrupted; durable progress was retained.", flush=True)
            return 130
        except (APIError, RuntimeError, ValueError) as exc:
            try:
                self._update_cohort(last_error=str(exc))
            except Exception:
                pass
            print(f"Sampled backfill stopped: {exc}", file=sys.stderr)
            return 1
