"""Read-only Phase 2 dataset audit for a frozen sampled cohort."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .labeling import GAMMA_RESOLUTION_SOURCE, HORIZONS

AUDIT_VERSION = 1
DEFAULT_MODEL_CATEGORIES = ("sports", "finance", "geopolitics", "politics")


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params).fetchall()]


def _row(connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    row = connection.execute(query, params).fetchone()
    if row is None:
        raise ValueError("Audit query unexpectedly returned no row")
    return dict(row)


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 4) if denominator else None


def audit_database(
    database: Path,
    cohort: str,
    *,
    categories: tuple[str, ...] = DEFAULT_MODEL_CATEGORIES,
    generated_at: int | None = None,
) -> dict[str, Any]:
    """Audit a cohort without migrating or writing to its SQLite database."""
    database = database.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    if not categories:
        raise ValueError("At least one modeling category is required")

    uri = f"{database.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cohort_row = _dict(connection.execute(
            "SELECT * FROM backfill_cohorts WHERE cohort_name=?", (cohort,),
        ).fetchone())
        if cohort_row is None:
            raise ValueError(f"Unknown cohort: {cohort}")

        checks: list[AuditCheck] = []

        def check(name: str, passed: bool, detail: str, severity: str = "error") -> None:
            checks.append(AuditCheck(name, bool(passed), detail, severity))

        quick_check = [row[0] for row in connection.execute("PRAGMA quick_check").fetchall()]
        foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
        check("sqlite_quick_check", quick_check == ["ok"],
              "result=" + ", ".join(str(value) for value in quick_check[:10]))
        check("sqlite_foreign_keys", not foreign_key_issues,
              f"violations={len(foreign_key_issues)}")

        check("cohort_complete", cohort_row["phase"] == "complete",
              f"phase={cohort_row['phase']}")
        check("discovery_frozen", cohort_row["discovery_complete"] == 1,
              f"discovery_complete={cohort_row['discovery_complete']}")
        check("gamma_resolution_contract",
              cohort_row["resolution_required"] == 1
              and cohort_row["resolution_source"] == GAMMA_RESOLUTION_SOURCE
              and cohort_row["resolution_status"] == "complete",
              "resolution_required={} source={} status={}".format(
                  cohort_row["resolution_required"], cohort_row["resolution_source"],
                  cohort_row["resolution_status"],
              ))

        market_status = _rows(connection, """SELECT canonical_category,fetch_status,COUNT(*) count
            FROM backfill_cohort_markets
            WHERE cohort_name=? AND selection_status='selected'
            GROUP BY canonical_category,fetch_status
            ORDER BY canonical_category,fetch_status""", (cohort,))
        selected = _row(connection, """SELECT COUNT(*) selected_markets,
                   COUNT(DISTINCT CASE WHEN COALESCE(event_id,event_slug,'')<>''
                         THEN COALESCE(event_id,event_slug) END) identified_events,
                   COALESCE(SUM(fetch_status='complete'),0) fetched_markets,
                   COALESCE(SUM(fetch_status='failed'),0) failed_markets,
                   COALESCE(SUM(fetch_status='pending'),0) pending_markets,
                   COALESCE(SUM(qualifying_trade_count),0) recorded_trade_count
            FROM backfill_cohort_markets
            WHERE cohort_name=? AND selection_status='selected'""", (cohort,))
        check("selected_market_scans_complete",
              selected["failed_markets"] == 0 and selected["pending_markets"] == 0
              and selected["fetched_markets"] == selected["selected_markets"],
              "selected={selected_markets} complete={fetched_markets} pending={pending_markets} "
              "failed={failed_markets}".format(**selected))

        resolutions = _row(connection, """SELECT COUNT(*) resolution_rows,
                   COALESCE(SUM(mr.terminal=1),0) terminal_rows,
                   COALESCE(SUM(mr.source=?),0) authoritative_rows,
                   COALESCE(SUM(mr.winning_outcome_index IS NULL),0) missing_winner_rows
            FROM backfill_cohort_markets cm
            LEFT JOIN market_resolutions mr ON mr.condition_id=cm.condition_id
            WHERE cm.cohort_name=? AND cm.selection_status='selected'
              AND mr.condition_id IS NOT NULL""", (GAMMA_RESOLUTION_SOURCE, cohort))
        resolution_ok = (
            resolutions["resolution_rows"] == selected["selected_markets"]
            and resolutions["terminal_rows"] == selected["selected_markets"]
            and resolutions["authoritative_rows"] == selected["selected_markets"]
            and resolutions["missing_winner_rows"] == 0
        )
        check("selected_markets_gamma_verified", resolution_ok,
              "selected={}; resolution_rows={resolution_rows}; terminal={terminal_rows}; "
              "authoritative={authoritative_rows}; missing_winner={missing_winner_rows}".format(
                  selected["selected_markets"], **resolutions))

        queue = _rows(connection, """SELECT status,job_kind,COUNT(*) count
            FROM jobs WHERE cohort_name IN (?,?)
            GROUP BY status,job_kind ORDER BY status,job_kind""", (cohort, f"{cohort}:live"))
        open_jobs = sum(row["count"] for row in queue if row["status"] in ("pending", "leased"))
        dead_jobs = sum(row["count"] for row in queue if row["status"] == "dead")
        audited_skips = _rows(connection, """SELECT job_kind,COUNT(*) count
            FROM jobs WHERE cohort_name IN (?,?) AND status='succeeded'
              AND last_error LIKE 'skipped%'
            GROUP BY job_kind ORDER BY job_kind""", (cohort, f"{cohort}:live"))
        dead_job_details = _rows(connection, """SELECT job_key,job_kind,entity_id,attempts,last_error
            FROM jobs WHERE cohort_name IN (?,?) AND status='dead'
            ORDER BY updated_at DESC,job_id DESC""", (cohort, f"{cohort}:live"))
        audited_skip_details = _rows(connection, """SELECT job_key,job_kind,entity_id,last_error
            FROM jobs WHERE cohort_name IN (?,?) AND status='succeeded'
              AND last_error LIKE 'skipped%'
            ORDER BY updated_at DESC,job_id DESC""", (cohort, f"{cohort}:live"))
        check("queue_drained", open_jobs == 0, f"pending_or_leased={open_jobs}")
        check("no_dead_jobs", dead_jobs == 0, f"dead={dead_jobs}")

        scope_params = (cohort, *categories)
        scope_marks = _placeholders(categories)
        trade_scope_from = """FROM backfill_cohort_markets cm
            JOIN backfill_cohorts c ON c.cohort_name=cm.cohort_name
            JOIN trades t ON t.condition_id=cm.condition_id
                         AND t.trade_ts BETWEEN c.window_start AND c.window_end"""
        trade_scope_where = f"""WHERE cm.cohort_name=? AND cm.selection_status='selected'
              AND cm.canonical_category IN ({scope_marks})"""

        population = _row(connection, f"""SELECT COUNT(*) trades,
                   COUNT(DISTINCT t.proxy_wallet) wallets,
                   COUNT(DISTINCT t.condition_id) markets,
                   COUNT(DISTINCT COALESCE(NULLIF(cm.event_id,''),NULLIF(cm.event_slug,''),cm.condition_id)) events,
                   MIN(t.trade_ts) first_trade_ts,MAX(t.trade_ts) last_trade_ts,
                   COALESCE(SUM(t.enrichment_status='complete'),0) enriched,
                   COALESCE(SUM(t.enrichment_status='pending'),0) enrichment_pending,
                   COALESCE(SUM(t.enrichment_status='failed'),0) enrichment_failed,
                   COALESCE(SUM(t.enrichment_status='not_selected'),0) enrichment_not_selected
            {trade_scope_from} {trade_scope_where}""", scope_params)
        check("modeling_scope_nonempty", population["trades"] > 0,
              f"categories={','.join(categories)} trades={population['trades']}")
        check("selected_enrichment_jobs_terminal",
              population["enrichment_pending"] == 0 and population["enrichment_failed"] == 0,
              f"pending={population['enrichment_pending']} failed={population['enrichment_failed']} "
              f"complete={population['enriched']} not_selected={population['enrichment_not_selected']}")

        duplicates = _row(connection, f"""SELECT
                   COUNT(*)-COUNT(DISTINCT t.trade_key) duplicate_trade_keys,
                   COALESCE(SUM(o.condition_id<>t.condition_id),0) outcome_condition_mismatches,
                   COALESCE(SUM(o.outcome_index IS NULL),0) missing_outcome_indexes
            {trade_scope_from}
            LEFT JOIN outcomes o ON o.asset_id=t.asset_id
            {trade_scope_where}""", scope_params)
        check("trade_keys_unique", duplicates["duplicate_trade_keys"] == 0,
              f"duplicates={duplicates['duplicate_trade_keys']}")
        check("trade_outcomes_aligned",
              duplicates["outcome_condition_mismatches"] == 0
              and duplicates["missing_outcome_indexes"] == 0,
              "condition_mismatches={outcome_condition_mismatches} "
              "missing_outcome_indexes={missing_outcome_indexes}".format(**duplicates))

        label_coverage: list[dict[str, Any]] = []
        for horizon in HORIZONS:
            labels = _row(connection, f"""SELECT COUNT(l.trade_key) label_rows,
                       COALESCE(SUM(l.status='complete'),0) complete_rows,
                       COALESCE(SUM(l.status='price_unavailable'),0) price_unavailable_rows,
                       COALESCE(SUM(l.status='unavailable_before_horizon'),0) before_horizon_rows,
                       COALESCE(SUM(l.status NOT IN ('complete','price_unavailable','unavailable_before_horizon')),0) other_status_rows,
                       COALESCE(SUM(l.status='complete' AND (l.entry_price_ppm IS NULL OR l.outcome_won IS NULL OR l.gross_pnl_microusd IS NULL)),0) incomplete_complete_rows,
                       COALESCE(SUM(l.scheduled_at<>t.trade_ts+l.horizon_seconds),0) scheduled_time_mismatches,
                       COALESCE(SUM(l.status='complete' AND (l.observed_at<l.scheduled_at OR l.observed_at>l.scheduled_at+300)),0) observation_time_mismatches,
                       COALESCE(SUM(l.status='complete' AND (l.entry_price_ppm<0 OR l.entry_price_ppm>1000000)),0) invalid_entry_price_rows,
                       COALESCE(SUM(l.status='complete' AND l.outcome_won<>CASE WHEN o.outcome_index=mr.winning_outcome_index THEN 1 ELSE 0 END),0) winner_mismatches,
                       COALESCE(SUM(l.status='complete' AND l.gross_pnl_microusd<>
                           CASE WHEN l.outcome_won=1 THEN 1000000-l.entry_price_ppm ELSE -l.entry_price_ppm END),0) pnl_mismatches
                {trade_scope_from}
                LEFT JOIN labels l ON l.trade_key=t.trade_key AND l.horizon_seconds=?
                LEFT JOIN outcomes o ON o.asset_id=t.asset_id
                LEFT JOIN market_resolutions mr ON mr.condition_id=t.condition_id
                {trade_scope_where}""", (horizon, *scope_params))
            labels["horizon_seconds"] = horizon
            labels["missing_rows"] = population["trades"] - labels["label_rows"]
            labels["coverage_percent"] = _pct(labels["label_rows"], population["trades"])
            labels["trainable_percent"] = _pct(labels["complete_rows"], population["trades"])
            label_coverage.append(labels)
            check(f"label_{horizon}_coverage", labels["missing_rows"] == 0,
                  f"rows={labels['label_rows']}/{population['trades']} missing={labels['missing_rows']}")
            check(f"label_{horizon}_integrity",
                  labels["other_status_rows"] == 0
                  and labels["incomplete_complete_rows"] == 0
                  and labels["scheduled_time_mismatches"] == 0
                  and labels["observation_time_mismatches"] == 0
                  and labels["invalid_entry_price_rows"] == 0
                  and labels["winner_mismatches"] == 0
                  and labels["pnl_mismatches"] == 0,
                  "other_status={other_status_rows} incomplete_complete={incomplete_complete_rows} "
                  "scheduled_time_mismatch={scheduled_time_mismatches} "
                  "observation_time_mismatch={observation_time_mismatches} "
                  "invalid_entry_price={invalid_entry_price_rows} "
                  "winner_mismatch={winner_mismatches} pnl_mismatch={pnl_mismatches}".format(**labels))

        label_sources = _rows(connection, f"""SELECT l.horizon_seconds,l.status,
                   COALESCE(l.price_source,'none') price_source,
                   COALESCE(l.price_quality,'none') price_quality,COUNT(*) rows
            {trade_scope_from}
            JOIN labels l ON l.trade_key=t.trade_key
            {trade_scope_where}
            GROUP BY l.horizon_seconds,l.status,COALESCE(l.price_source,'none'),
                     COALESCE(l.price_quality,'none')
            ORDER BY l.horizon_seconds,l.status,price_source,price_quality""", scope_params)

        feature_coverage = _row(connection, f"""SELECT
                   COUNT(v.trade_key) feature_rows,
                   COALESCE(SUM(v.wallet_completeness='complete'),0) complete_wallet_rows,
                   COALESCE(SUM(v.wallet_completeness='partial'),0) partial_wallet_rows,
                   COALESCE(SUM(v.wallet_completeness='unavailable'),0) unavailable_wallet_rows,
                   COALESCE(SUM(v.wallet_coverage_details LIKE '%truncated%'),0) truncated_wallet_rows,
                   COALESCE(SUM(v.size_vs_median_milli IS NULL),0) missing_size_ratio_rows,
                   COALESCE(SUM(v.seconds_to_resolution IS NULL),0) missing_resolution_time_rows
            {trade_scope_from}
            LEFT JOIN v_model_features v ON v.trade_key=t.trade_key
            {trade_scope_where}""", scope_params)
        feature_coverage["coverage_percent"] = _pct(feature_coverage["feature_rows"], population["trades"])
        check("completed_enrichments_have_features",
              feature_coverage["feature_rows"] == population["enriched"],
              f"feature_rows={feature_coverage['feature_rows']} enrichment_complete={population['enriched']}")

        categories_report = _rows(connection, f"""SELECT cm.canonical_category,
                   COUNT(*) trades,COUNT(DISTINCT t.proxy_wallet) wallets,
                   COUNT(DISTINCT t.condition_id) markets,
                   COUNT(DISTINCT COALESCE(NULLIF(cm.event_id,''),NULLIF(cm.event_slug,''),cm.condition_id)) events,
                   ROUND(AVG(t.price_ppm)/1000000.0,6) mean_trade_price
            {trade_scope_from} {trade_scope_where}
            GROUP BY cm.canonical_category ORDER BY cm.canonical_category""", scope_params)
        class_balance = _rows(connection, f"""SELECT cm.canonical_category,l.outcome_won,COUNT(*) trades,
                   COUNT(DISTINCT t.condition_id) markets
            {trade_scope_from}
            JOIN labels l ON l.trade_key=t.trade_key AND l.horizon_seconds=900 AND l.status='complete'
            {trade_scope_where}
            GROUP BY cm.canonical_category,l.outcome_won
            ORDER BY cm.canonical_category,l.outcome_won""", scope_params)
        price_bands = _rows(connection, f"""SELECT CASE
                   WHEN t.price_ppm<100000 THEN '00-10%'
                   WHEN t.price_ppm<250000 THEN '10-25%'
                   WHEN t.price_ppm<500000 THEN '25-50%'
                   WHEN t.price_ppm<750000 THEN '50-75%'
                   WHEN t.price_ppm<900000 THEN '75-90%'
                   ELSE '90-100%' END price_band,COUNT(*) trades
            {trade_scope_from} {trade_scope_where}
            GROUP BY price_band ORDER BY MIN(t.price_ppm)""", scope_params)
        top_markets = _rows(connection, f"""SELECT t.condition_id,cm.canonical_category,
                   cm.title,COUNT(*) trades,COUNT(DISTINCT t.proxy_wallet) wallets
            {trade_scope_from} {trade_scope_where}
            GROUP BY t.condition_id,cm.canonical_category,cm.title
            ORDER BY trades DESC,t.condition_id LIMIT 20""", scope_params)
        top_wallets = _rows(connection, f"""SELECT t.proxy_wallet,COUNT(*) trades,
                   COUNT(DISTINCT t.condition_id) markets
            {trade_scope_from} {trade_scope_where}
            GROUP BY t.proxy_wallet ORDER BY trades DESC,t.proxy_wallet LIMIT 20""", scope_params)
        max_market_share = _pct(top_markets[0]["trades"], population["trades"]) if top_markets else None
        max_wallet_share = _pct(top_wallets[0]["trades"], population["trades"]) if top_wallets else None

        report = {
            "audit_version": AUDIT_VERSION,
            "generated_at": int(time.time()) if generated_at is None else generated_at,
            "database": str(database),
            "cohort": cohort,
            "modeling_categories": list(categories),
            "passed": all(item.passed or item.severity != "error" for item in checks),
            "checks": [item.as_dict() for item in checks],
            "cohort_state": cohort_row,
            "selected_markets": {**selected, "by_category_and_fetch_status": market_status},
            "resolutions": resolutions,
            "queue": {
                "by_status_and_kind": queue,
                "open_jobs": open_jobs,
                "dead_jobs": dead_jobs,
                "audited_skips": audited_skips,
                "dead_job_details": dead_job_details,
                "audited_skip_details": audited_skip_details,
            },
            "population": population,
            "label_coverage": label_coverage,
            "label_source_quality": label_sources,
            "feature_coverage": feature_coverage,
            "integrity": duplicates,
            "category_summary": categories_report,
            "winner_balance_900s": class_balance,
            "observed_trade_price_bands": price_bands,
            "concentration": {
                "largest_market_trade_share_percent": max_market_share,
                "largest_wallet_trade_share_percent": max_wallet_share,
                "top_markets": top_markets,
                "top_wallets": top_wallets,
            },
        }
        return report
    except sqlite3.Error as exc:
        raise ValueError(f"Could not audit database schema: {exc}") from exc
    finally:
        connection.close()


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact human-readable companion to the JSON audit."""
    status = "PASS" if report["passed"] else "FAIL"
    population = report["population"]
    features = report["feature_coverage"]
    lines = [
        f"# Phase 2 dataset audit: {report['cohort']}", "",
        f"**Overall status: {status}**", "",
        f"Database: `{report['database']}`", "",
        f"Modeling categories: {', '.join(report['modeling_categories'])}", "",
        "## Population", "",
        "| Trades | Wallets | Markets | Events | Feature rows |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {population['trades']} | {population['wallets']} | {population['markets']} | "
        f"{population['events']} | {features['feature_rows']} |", "",
        "## Hard checks", "",
        "| Result | Check | Detail |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        result = "PASS" if item["passed"] else "FAIL"
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {result} | `{item['name']}` | {detail} |")
    lines.extend(["", "## Label coverage", "",
                  "| Horizon | Rows | Missing | Complete/trainable | Price unavailable | Before horizon |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in report["label_coverage"]:
        lines.append(
            f"| {item['horizon_seconds']}s | {item['label_rows']} | {item['missing_rows']} | "
            f"{item['complete_rows']} | {item['price_unavailable_rows']} | {item['before_horizon_rows']} |"
        )
    lines.extend(["", "## Enrichment and wallet coverage", "",
                  "| Feature rows | Complete history | Partial | Unavailable | Truncated |",
                  "| ---: | ---: | ---: | ---: | ---: |",
                  f"| {features['feature_rows']} | {features['complete_wallet_rows']} | "
                  f"{features['partial_wallet_rows']} | {features['unavailable_wallet_rows']} | "
                  f"{features['truncated_wallet_rows']} |", "",
                  "## Category summary", "",
                  "| Category | Trades | Wallets | Markets | Events | Mean trade price |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for item in report["category_summary"]:
        lines.append(
            f"| {item['canonical_category']} | {item['trades']} | {item['wallets']} | "
            f"{item['markets']} | {item['events']} | {item['mean_trade_price']} |"
        )
    concentration = report["concentration"]
    lines.extend(["", "## Concentration", "",
                  f"Largest market share of trades: {concentration['largest_market_trade_share_percent']}%  ",
                  f"Largest wallet share of trades: {concentration['largest_wallet_trade_share_percent']}%", "",
                  "The JSON report contains the full status distributions, winner balance, price bands, "
                  "and top market/wallet tables.", ""])
    return "\n".join(lines)


def write_audit(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{report['cohort']}_audit"
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
