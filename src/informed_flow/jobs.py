"""Durable, at-least-once SQLite work queue."""
from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .core import canonical_json
from .db import Database


@dataclass(frozen=True)
class Job:
    job_id: int
    job_key: str
    job_kind: str
    cohort_name: str | None
    entity_id: str | None
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


class DeferredJob(RuntimeError):
    """A recurring job should become pending again without counting as failure."""

    def __init__(self, available_at: int):
        super().__init__(f"deferred until {available_at}")
        self.available_at = available_at


class JobQueue:
    def __init__(self, db: Database, *, clock=time.time):
        self.db = db
        self.clock = clock

    def enqueue(
        self,
        kind: str,
        key: str,
        payload: Mapping[str, Any],
        *,
        cohort: str | None = None,
        entity_id: str | None = None,
        priority: int = 0,
        available_at: int | None = None,
        max_attempts: int = 8,
        reopen_succeeded: bool = False,
    ) -> bool:
        now = int(self.clock())
        with self.db.connection:
            cursor = self.enqueue_in_transaction(
                self.db.connection, kind, key, payload, now=now, cohort=cohort,
                entity_id=entity_id, priority=priority, available_at=available_at,
                max_attempts=max_attempts,
            )
        if cursor.rowcount or not reopen_succeeded:
            return bool(cursor.rowcount)
        with self.db.connection:
            reopened = self.db.connection.execute(
                """UPDATE jobs SET status='pending',available_at=?,attempts=0,last_error=NULL,
                          finished_at=NULL,updated_at=?
                   WHERE job_key=? AND status='succeeded'""",
                (available_at if available_at is not None else now, now, key),
            )
        return bool(reopened.rowcount)

    @staticmethod
    def enqueue_in_transaction(
        connection: sqlite3.Connection,
        kind: str,
        key: str,
        payload: Mapping[str, Any],
        *,
        now: int,
        cohort: str | None = None,
        entity_id: str | None = None,
        priority: int = 0,
        available_at: int | None = None,
        max_attempts: int = 8,
    ) -> sqlite3.Cursor:
        """Insert a job using the caller's transaction."""
        return connection.execute(
            """INSERT OR IGNORE INTO jobs(
                   job_key,job_kind,cohort_name,entity_id,payload_json,priority,status,
                   available_at,max_attempts,created_at,updated_at
               ) VALUES (?,?,?,?,?,?,'pending',?,?,?,?)""",
            (
                key, kind, cohort, entity_id, canonical_json(payload), priority,
                available_at if available_at is not None else now,
                max_attempts, now, now,
            ),
        )

    def recover_expired(self, cohort: str | None = None) -> int:
        now = int(self.clock())
        sql = """UPDATE jobs SET status='pending',lease_owner=NULL,lease_expires_at=NULL,
                          updated_at=?
                 WHERE status='leased' AND lease_expires_at<=?"""
        params: tuple[Any, ...] = (now, now)
        if cohort is not None:
            sql += " AND cohort_name=?"
            params += (cohort,)
        with self.db.connection:
            cursor = self.db.connection.execute(sql, params)
        return cursor.rowcount

    def claim(
        self,
        worker: str,
        kinds: Iterable[str],
        *,
        cohort: str | None = None,
        lease_seconds: int = 300,
    ) -> Job | None:
        kinds = tuple(kinds)
        if not kinds:
            return None
        now = int(self.clock())
        placeholders = ",".join("?" for _ in kinds)
        cohort_clause = " AND cohort_name=?" if cohort is not None else ""
        params: tuple[Any, ...] = (now, *kinds)
        if cohort is not None:
            params += (cohort,)
        connection = self.db.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""SELECT * FROM jobs
                    WHERE status='pending' AND available_at<=?
                      AND job_kind IN ({placeholders})
                      {cohort_clause}
                    ORDER BY priority DESC,available_at,job_id LIMIT 1""",
                params,
            ).fetchone()
            if not row:
                connection.commit()
                return None
            cursor = connection.execute(
                """UPDATE jobs SET status='leased',lease_owner=?,lease_expires_at=?,
                          attempts=attempts+1,updated_at=?
                   WHERE job_id=? AND status='pending'""",
                (worker, now + lease_seconds, now, row["job_id"]),
            )
            connection.commit()
            if cursor.rowcount != 1:
                return None
            return Job(
                int(row["job_id"]), row["job_key"], row["job_kind"],
                row["cohort_name"], row["entity_id"], json.loads(row["payload_json"]),
                int(row["attempts"]) + 1, int(row["max_attempts"]),
            )
        except BaseException:
            connection.rollback()
            raise

    def renew(self, job: Job, worker: str, *, lease_seconds: int = 300) -> None:
        now = int(self.clock())
        with self.db.connection:
            self.db.connection.execute(
                """UPDATE jobs SET lease_expires_at=?,updated_at=?
                   WHERE job_id=? AND status='leased' AND lease_owner=?""",
                (now + lease_seconds, now, job.job_id, worker),
            )

    def succeed(self, job: Job, worker: str) -> None:
        now = int(self.clock())
        with self.db.connection:
            self.db.connection.execute(
                """UPDATE jobs SET status='succeeded',lease_owner=NULL,lease_expires_at=NULL,
                          last_error=NULL,finished_at=?,updated_at=?
                   WHERE job_id=? AND status='leased' AND lease_owner=?""",
                (now, now, job.job_id, worker),
            )

    def defer(self, job: Job, worker: str, available_at: int) -> None:
        now = int(self.clock())
        with self.db.connection:
            self.db.connection.execute(
                """UPDATE jobs SET status='pending',available_at=?,lease_owner=NULL,
                          lease_expires_at=NULL,attempts=MAX(0,attempts-1),updated_at=?
                   WHERE job_id=? AND status='leased' AND lease_owner=?""",
                (available_at, now, job.job_id, worker),
            )

    def fail(self, job: Job, worker: str, error: BaseException, *, retryable: bool = True) -> str:
        now = int(self.clock())
        dead = not retryable or job.attempts >= job.max_attempts
        status = "dead" if dead else "pending"
        delay = min(6 * 3600, 60 * (2 ** max(0, job.attempts - 1)))
        available = now if dead else now + delay + random.randint(0, min(30, delay))
        with self.db.connection:
            self.db.connection.execute(
                """UPDATE jobs SET status=?,available_at=?,lease_owner=NULL,
                          lease_expires_at=NULL,last_error=?,finished_at=?,updated_at=?
                   WHERE job_id=? AND status='leased' AND lease_owner=?""",
                (
                    status, available, str(error) or type(error).__name__,
                    now if dead else None, now, job.job_id, worker,
                ),
            )
        return status

    def counts(self, cohort: str | None = None) -> dict[str, int]:
        sql = "SELECT status,COUNT(*) n FROM jobs"
        params: tuple[Any, ...] = ()
        if cohort is not None:
            sql += " WHERE cohort_name=?"
            params = (cohort,)
        sql += " GROUP BY status"
        return {row["status"]: row["n"] for row in self.db.rows(sql, params)}

    def retry_dead(self, *, kind: str | None = None, entity_id: str | None = None) -> int:
        clauses = ["status='dead'"]
        params: list[Any] = []
        if kind:
            clauses.append("job_kind=?")
            params.append(kind)
        if entity_id:
            clauses.append("entity_id=?")
            params.append(entity_id)
        now = int(self.clock())
        with self.db.connection:
            cursor = self.db.connection.execute(
                f"""UPDATE jobs SET status='pending',available_at=?,attempts=0,last_error=NULL,
                            finished_at=NULL,updated_at=? WHERE {' AND '.join(clauses)}""",
                (now, now, *params),
            )
        return cursor.rowcount
