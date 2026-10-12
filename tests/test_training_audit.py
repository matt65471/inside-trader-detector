from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from informed_flow.db import Database
from informed_flow.training_audit import audit_database, render_markdown, write_audit


class TrainingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "audit.sqlite3"
        self.db = Database(self.path)
        self.db.initialize()
        with self.db.connection:
            self.db.connection.execute(
                """INSERT INTO backfill_cohorts(
                       cohort_name,window_start,window_end,days,category_limit,seed,
                       embedding_model,similarity_threshold_ppm,requested_device,effective_device,
                       history_mode,admission_rate_ppm,discovery_strategy,resolution_required,
                       resolution_source,resolution_status,phase,discovery_complete,created_at,updated_at
                   ) VALUES ('cohort',1,2000000,0,1,0,'test',900000,'cpu','cpu','lifetime',
                             500000,'tag_filtered_v2',1,'gamma_markets_v1','complete','complete',1,1,1)"""
            )
            self.db.connection.execute(
                """INSERT INTO markets(condition_id,event_id,title,slug,event_slug,category,end_ts,
                       first_seen_at,last_refreshed_at)
                   VALUES ('condition','event','Question?','question','event','finance',1900000,1,1)"""
            )
            self.db.connection.execute(
                "INSERT INTO outcomes VALUES ('asset','condition','Yes',0,1)"
            )
            self.db.connection.execute(
                """INSERT INTO backfill_cohort_markets(
                       cohort_name,condition_id,event_id,event_slug,title,canonical_category,raw_json,
                       eligible,random_rank,selection_status,selection_reason,fetch_status,
                       qualifying_trade_count,discovered_at)
                   VALUES ('cohort','condition','event','event','Question?','finance','{}',1,'1',
                           'selected','seeded_admission','complete',1,1)"""
            )
            self.db.connection.execute(
                """INSERT INTO market_resolutions VALUES(
                       'condition','resolved',1,0,'[1,0]',1900000,0,'gamma_markets_v1',
                       'terminal_outcome_prices','{}',1,NULL)"""
            )
            self.db.connection.execute(
                """INSERT INTO trades VALUES(
                       'trade','source','fingerprint','transaction','wallet','condition','asset','BUY',
                       400000,1000000,400000,1000000,1000001,'backfill','{}','complete')"""
            )
            self.db.connection.execute(
                """INSERT INTO wallet_snapshots(
                       proxy_wallet,as_of_ts,feature_version,prior_trade_count,prior_market_count,
                       mean_notional_microusd,median_notional_microusd,max_notional_microusd,
                       completeness,missing_reason,computed_at)
                   VALUES ('wallet',1000000,2,2,1,300000,300000,400000,'complete',
                           'history_window_complete',1)"""
            )
            snapshot_id = self.db.connection.execute(
                "SELECT wallet_snapshot_id FROM wallet_snapshots"
            ).fetchone()[0]
            self.db.connection.execute(
                """INSERT INTO trade_features VALUES(
                       'trade',2,?,NULL,1333,900000,400000,0,0,0,'high',1,'[]')""",
                (snapshot_id,),
            )
            for horizon in (900, 3600, 86400):
                self.db.connection.execute(
                    """INSERT INTO labels(
                           trade_key,horizon_seconds,scheduled_at,observed_at,entry_price_ppm,
                           price_quality,outcome_won,status,price_source,gross_pnl_microusd,source_json)
                       VALUES ('trade',?,1000000+?,1000001+?,450000,'exact',1,'complete',
                               'clob_price_history',550000,'{}')""",
                    (horizon, horizon, horizon),
                )

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def test_complete_cohort_passes_and_writes_reports(self) -> None:
        self.db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        report = audit_database(self.path, "cohort", generated_at=123)
        self.assertTrue(report["passed"])
        self.assertEqual(report["population"]["trades"], 1)
        self.assertIn("Overall status: PASS", render_markdown(report))
        output = Path(self.tempdir.name) / "output"
        json_path, markdown_path = write_audit(report, output)
        self.assertTrue(markdown_path.exists())
        self.assertEqual(json.loads(json_path.read_text())["generated_at"], 123)

    def test_missing_label_fails_without_mutating_database(self) -> None:
        with self.db.connection:
            self.db.connection.execute(
                "DELETE FROM labels WHERE horizon_seconds=900"
            )
        before = self.path.stat().st_mtime_ns
        report = audit_database(self.path, "cohort", generated_at=123)
        after = self.path.stat().st_mtime_ns
        self.assertFalse(report["passed"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("label_900_coverage", failed)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
