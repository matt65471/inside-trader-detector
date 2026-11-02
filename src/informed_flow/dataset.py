"""Build a leakage-safe, time-split training dataset from a resolved cohort.

The database is opened read-only. Features and labels are written to separate
files, and test labels go to their own file so they can stay sealed until a
final evaluation. Nothing is trained here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import FEATURE_VERSION, SCORE_VERSION

DATASET_VERSION = 1
DEFAULT_CATEGORIES = ("sports", "finance", "geopolitics", "politics", "culture")
GAMMA_SOURCE = "gamma_markets_v1"
HORIZONS = ((900, "15m"), (3600, "1h"), (86400, "24h"))
BURST_GAP_SECONDS = 60
CV_FOLDS = 5
# Enrichment keeps every trade >= $500 and a deterministic 1-in-4 sample below it.
ENRICH_NOTIONAL_MICROUSD = 500_000_000
SMALL_TRADE_INCLUSION = 0.25
MIN_MARKETS_PER_CELL = 30

# Stored fields deliberately left out of features.csv, with the reason.
EXCLUDED_FIELDS = {
    "market_snapshots.*": "latest snapshot was fetched after the market closed (last price, liquidity, volume)",
    "markets.closed / accepting_orders": "describe the market after resolution",
    "market_resolutions.*": "the outcome being predicted",
    "labels.*": "outcomes and post-trade prices; written only to labels files",
    "young_wallet_buys_1h / wallet_age_seconds": "always unknown in this dataset",
    "order_book_snapshots.*": "historical books are unavailable for every trade",
    "ingested_at / computed_at": "collection-process timestamps, not market information",
    "heuristic_score / heuristic_bucket": "kept as a baseline column, not a model feature",
}

WALLET_FIELDS = (
    ("wallet_observed_trade_count", "observed_trade_count", 1),
    ("wallet_distinct_market_count", "distinct_market_count", 1),
    ("wallet_distinct_outcome_count", "distinct_outcome_count", 1),
    ("wallet_active_days", "active_days", 1),
    ("wallet_mean_notional_usd", "mean_notional_microusd", 1e-6),
    ("wallet_median_notional_usd", "median_notional_microusd", 1e-6),
    ("wallet_p90_notional_usd", "p90_notional_microusd", 1e-6),
    ("wallet_max_notional_usd", "max_notional_microusd", 1e-6),
    ("wallet_stddev_notional_usd", "stddev_notional_microusd", 1e-6),
    ("wallet_mean_price", "mean_price_ppm", 1e-6),
    ("wallet_cheap_trade_count", "cheap_trade_count", 1),
    ("wallet_last_7d_trade_count", "last_7d_trade_count", 1),
    ("wallet_last_7d_notional_usd", "last_7d_notional_microusd", 1e-6),
    ("wallet_seconds_since_last_trade", "seconds_since_last_trade", 1),
    ("wallet_largest_market_volume_share", "largest_market_volume_share_ppm", 1e-6),
)

OPTIONAL_FEATURES = (
    "days_since_market_start", "hours_to_scheduled_end", "fraction_of_market_life_elapsed",
    "size_vs_median", *(name for name, _, _ in WALLET_FIELDS),
    "wallet_buy_count", "wallet_sell_count", "wallet_buy_notional_share",
)

FEATURE_COLUMNS = (
    "trade_key", "condition_id", "event_slug", "proxy_wallet", "canonical_category",
    "split", "split_basis", "cv_fold",
    "price", "notional_usd", "log_notional", "size_shares", "outcome_name",
    "trade_ts", "trade_time_utc", "hour_utc", "day_of_week",
    "days_since_market_start", "hours_to_scheduled_end", "fraction_of_market_life_elapsed",
    "months_since_dataset_start",
    *(name for name, _, _ in WALLET_FIELDS),
    "wallet_buy_count", "wallet_sell_count", "wallet_buy_notional_share",
    "size_vs_median", "history_truncated",
    "same_asset_buys_1h", "same_asset_notional_1h_usd",
    "burst_id", "burst_size",
    "heuristic_score", "heuristic_bucket",
    *(f"missing_{name}" for name in OPTIONAL_FEATURES),
    "market_weight", "sample_weight",
)

LABEL_COLUMNS = (
    "trade_key", "split", "outcome_won", "edge",
    *(f"{column}_{suffix}" for _, suffix in HORIZONS
      for column in ("entry_price", "gross_pnl_per_share", "label_status", "price_source")),
)


@dataclass(frozen=True)
class DatasetConfig:
    db_path: Path
    cohort: str
    output: Path
    test_start: int
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    seed: int = 0


def _hash_int(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def open_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_rows(
    connection: sqlite3.Connection, cohort: str, categories: Sequence[str],
) -> tuple[list[dict[str, Any]], Counter]:
    """Return enriched, resolved cohort trades in the chosen categories, plus drop counts."""
    if not connection.execute(
        "SELECT 1 FROM backfill_cohorts WHERE cohort_name=?", (cohort,)
    ).fetchone():
        raise ValueError(f"Unknown cohort: {cohort}")
    rows = connection.execute(
        """SELECT t.trade_key,t.condition_id,t.asset_id,t.proxy_wallet,t.price_ppm,
                  t.size_microshares,t.notional_microusd,t.trade_ts,
                  m.event_slug,m.start_ts,m.end_ts,o.outcome_name,cm.canonical_category,
                  r.resolved_at,r.terminal AS resolution_terminal,
                  tf.trade_key IS NOT NULL AS has_features,
                  tf.size_vs_median_milli,tf.same_asset_buys_1h,tf.same_asset_notional_1h,
                  ws.missing_reason AS wallet_reason,s.score,s.bucket
           FROM backfill_cohort_markets cm
           JOIN backfill_cohorts c ON c.cohort_name=cm.cohort_name
           JOIN trades t ON t.condition_id=cm.condition_id
                AND t.trade_ts BETWEEN c.window_start AND c.window_end
           JOIN markets m ON m.condition_id=t.condition_id
           JOIN outcomes o ON o.asset_id=t.asset_id
           LEFT JOIN market_resolutions r ON r.condition_id=t.condition_id AND r.source=?
           LEFT JOIN trade_features tf ON tf.trade_key=t.trade_key AND tf.feature_version=?
           LEFT JOIN wallet_snapshots ws ON ws.wallet_snapshot_id=tf.wallet_snapshot_id
           LEFT JOIN scores s ON s.trade_key=t.trade_key AND s.score_version=?
           WHERE cm.cohort_name=? AND cm.selection_status='selected'
           ORDER BY t.trade_ts,t.trade_key""",
        (GAMMA_SOURCE, FEATURE_VERSION, SCORE_VERSION, cohort),
    ).fetchall()
    wanted = set(categories)
    kept: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for row in rows:
        if row["canonical_category"] not in wanted:
            dropped["other_category"] += 1
        elif not row["resolution_terminal"]:
            dropped["no_verified_resolution"] += 1
        elif not row["has_features"]:
            dropped["not_enriched"] += 1
        else:
            kept.append(dict(row))
    return kept, dropped


def _fetch_id(reason: str | None) -> int | None:
    for part in (reason or "").split(";"):
        prefix, _, value = part.partition(":")
        if prefix in ("history_fetch_id", "raw_history_fetch_id") and value.isdigit():
            return int(value)
    return None


def load_wallet_summaries(
    connection: sqlite3.Connection, rows: Iterable[Mapping[str, Any]],
) -> dict[int, tuple[str, dict[str, Any]]]:
    ids = sorted({fid for row in rows if (fid := _fetch_id(row["wallet_reason"])) is not None})
    found: dict[int, tuple[str, dict[str, Any]]] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for item in connection.execute(
            f"""SELECT h.fetch_id,h.status,s.summary_json FROM wallet_history_fetches h
                JOIN wallet_history_summaries s USING(fetch_id) WHERE h.fetch_id IN ({marks})""",
            chunk,
        ):
            found[int(item["fetch_id"])] = (item["status"], json.loads(item["summary_json"]))
    return found


def load_labels(
    connection: sqlite3.Connection, trade_keys: Iterable[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    keys = sorted(set(trade_keys))
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for start in range(0, len(keys), 500):
        chunk = keys[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for item in connection.execute(
            f"""SELECT trade_key,horizon_seconds,status,outcome_won,entry_price_ppm,
                       gross_pnl_microusd,price_source
                FROM labels WHERE trade_key IN ({marks})""",
            chunk,
        ):
            found[(item["trade_key"], int(item["horizon_seconds"]))] = dict(item)
    return found


def assign_splits(
    rows: Sequence[Mapping[str, Any]], test_start: int, seed: int,
) -> dict[str, tuple[str, str, int | None]]:
    """Map condition_id -> (split, split_basis, cv_fold). Every market lands in one split.

    test: first trade on/after the cutoff. train: resolved before the cutoff.
    Otherwise the market overlaps the cutoff and a seeded coin flip assigns it whole.
    """
    first_trade: dict[str, int] = {}
    resolved: dict[str, int | None] = {}
    for row in rows:
        market = row["condition_id"]
        first_trade[market] = min(first_trade.get(market, row["trade_ts"]), row["trade_ts"])
        resolved[market] = row["resolved_at"]
    result: dict[str, tuple[str, str, int | None]] = {}
    for market in sorted(first_trade):
        resolved_at = resolved[market]
        if first_trade[market] >= test_start:
            result[market] = ("test", "after_cutoff", None)
        elif resolved_at is not None and resolved_at < test_start:
            result[market] = ("train", "before_cutoff", None)
        else:
            side = "train" if _hash_int(seed, "overlap", market) % 2 == 0 else "test"
            result[market] = (side, "overlap_random", None)
    train = sorted(
        (market for market, value in result.items() if value[0] == "train"),
        key=lambda market: (_hash_int(seed, "fold", market), market),
    )
    for index, market in enumerate(train):
        split, basis, _ = result[market]
        result[market] = (split, basis, index % CV_FOLDS)
    return result


def assign_bursts(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, int]]:
    """Group a wallet's buys of one asset that are within 60s of the previous buy."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["proxy_wallet"], row["asset_id"])].append(row)
    result: dict[str, tuple[str, int]] = {}
    for trades in groups.values():
        trades = sorted(trades, key=lambda row: (row["trade_ts"], row["trade_key"]))
        bursts: list[list[Mapping[str, Any]]] = []
        for trade in trades:
            if bursts and trade["trade_ts"] - bursts[-1][-1]["trade_ts"] <= BURST_GAP_SECONDS:
                bursts[-1].append(trade)
            else:
                bursts.append([trade])
        for burst in bursts:
            burst_id = burst[0]["trade_key"]
            for trade in burst:
                result[trade["trade_key"]] = (burst_id, len(burst))
    return result


def sample_weight(notional_microusd: int) -> float:
    return 1.0 if notional_microusd >= ENRICH_NOTIONAL_MICROUSD else 1 / SMALL_TRADE_INCLUSION


def _scaled(value: Any, factor: float) -> float | int | None:
    if value is None:
        return None
    return value if factor == 1 else value * factor


def build_features(
    rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, tuple[str, str, int | None]],
    summaries: Mapping[int, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    bursts = assign_bursts(rows)
    per_market = Counter(row["condition_id"] for row in rows)
    dataset_start = min((row["trade_ts"] for row in rows), default=0)
    output: list[dict[str, Any]] = []
    for row in rows:
        ts = row["trade_ts"]
        moment = datetime.fromtimestamp(ts, tz=timezone.utc)
        start, end = row["start_ts"], row["end_ts"]
        split, basis, fold = splits[row["condition_id"]]
        fetch = summaries.get(_fetch_id(row["wallet_reason"]))
        status, summary = fetch if fetch else (None, {})
        notional = row["notional_microusd"]
        feature: dict[str, Any] = {
            "trade_key": row["trade_key"], "condition_id": row["condition_id"],
            "event_slug": row["event_slug"], "proxy_wallet": row["proxy_wallet"],
            "canonical_category": row["canonical_category"],
            "split": split, "split_basis": basis, "cv_fold": fold,
            "price": row["price_ppm"] / 1e6,
            "notional_usd": notional / 1e6,
            "log_notional": math.log10(notional / 1e6) if notional > 0 else None,
            "size_shares": row["size_microshares"] / 1e6,
            "outcome_name": row["outcome_name"],
            "trade_ts": ts, "trade_time_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hour_utc": moment.hour, "day_of_week": moment.weekday(),
            "days_since_market_start": (ts - start) / 86400 if start is not None else None,
            "hours_to_scheduled_end": (end - ts) / 3600 if end is not None else None,
            "fraction_of_market_life_elapsed": (
                (ts - start) / (end - start)
                if start is not None and end is not None and end > start else None
            ),
            "months_since_dataset_start": (ts - dataset_start) / (30.4375 * 86400),
        }
        for name, key, factor in WALLET_FIELDS:
            feature[name] = _scaled(summary.get(key), factor)
        sides = summary.get("side_counts") or {}
        side_volume = summary.get("side_notional_microusd") or {}
        total_volume = sum(side_volume.values())
        feature["wallet_buy_count"] = sides.get("BUY", 0) if summary else None
        feature["wallet_sell_count"] = sides.get("SELL", 0) if summary else None
        feature["wallet_buy_notional_share"] = (
            side_volume.get("BUY", 0) / total_volume if total_volume else None
        )
        size_ratio = row["size_vs_median_milli"]
        feature["size_vs_median"] = size_ratio / 1000 if size_ratio is not None else None
        feature["history_truncated"] = None if status is None else int(status == "truncated")
        feature["same_asset_buys_1h"] = row["same_asset_buys_1h"]
        feature["same_asset_notional_1h_usd"] = _scaled(row["same_asset_notional_1h"], 1e-6)
        feature["burst_id"], feature["burst_size"] = bursts[row["trade_key"]]
        feature["heuristic_score"] = row["score"]
        feature["heuristic_bucket"] = row["bucket"]
        for name in OPTIONAL_FEATURES:
            feature[f"missing_{name}"] = int(feature[name] is None)
        feature["market_weight"] = 1 / per_market[row["condition_id"]]
        feature["sample_weight"] = sample_weight(notional)
        output.append(feature)
    return output


def build_labels(
    rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, tuple[str, str, int | None]],
    labels: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        key = row["trade_key"]
        won = next(
            (labels[(key, seconds)]["outcome_won"] for seconds, _ in HORIZONS
             if (key, seconds) in labels and labels[(key, seconds)]["outcome_won"] is not None),
            None,
        )
        record: dict[str, Any] = {
            "trade_key": key, "split": splits[row["condition_id"]][0], "outcome_won": won,
            "edge": won - row["price_ppm"] / 1e6 if won is not None else None,
        }
        for seconds, suffix in HORIZONS:
            label = labels.get((key, seconds)) or {}
            record[f"entry_price_{suffix}"] = _scaled(label.get("entry_price_ppm"), 1e-6)
            record[f"gross_pnl_per_share_{suffix}"] = _scaled(label.get("gross_pnl_microusd"), 1e-6)
            record[f"label_status_{suffix}"] = label.get("status", "missing")
            record[f"price_source_{suffix}"] = label.get("price_source")
        output.append(record)
    return output


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 9))
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_table(
    features: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for feature in features:
        cell = cells.setdefault(
            (feature["split"], feature["canonical_category"]),
            {"markets": set(), "trades": 0, "won": 0, "known": 0, "price": 0.0},
        )
        cell["markets"].add(feature["condition_id"])
        cell["trades"] += 1
        cell["price"] += feature["price"]
        # Test outcomes stay sealed: never summarise them in the report or manifest.
        won = labels[feature["trade_key"]]["outcome_won"] if feature["split"] == "train" else None
        if won is not None:
            cell["known"] += 1
            cell["won"] += won
    return [
        {
            "split": split, "category": category, "markets": len(cell["markets"]),
            "trades": cell["trades"],
            "win_rate": cell["won"] / cell["known"] if cell["known"] else None,
            "mean_price": cell["price"] / cell["trades"],
        }
        for (split, category), cell in sorted(cells.items())
    ]


def _report(
    config: DatasetConfig, dropped: Counter, features: Sequence[Mapping[str, Any]],
    table: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]],
) -> str:
    total = len(features)
    lines = [
        f"# Dataset report (version {DATASET_VERSION})", "",
        f"- Cohort: `{config.cohort}`; categories: {', '.join(config.categories)}",
        f"- Test cutoff: {datetime.fromtimestamp(config.test_start, tz=timezone.utc):%Y-%m-%d %H:%M} UTC; seed {config.seed}",
        f"- Rows kept: {total}", "",
        "## Rows not included", "",
    ]
    for reason in ("other_category", "no_verified_resolution", "not_enriched"):
        lines.append(f"- {reason}: {dropped.get(reason, 0)}")
    basis = Counter((row["split"], row["split_basis"]) for row in features)
    markets = defaultdict(set)
    for row in features:
        markets[(row["split"], row["split_basis"])].add(row["condition_id"])
    lines += ["", "## Split assignment", "", "| split | basis | markets | trades |", "|---|---|---|---|"]
    for key in sorted(basis):
        lines.append(f"| {key[0]} | {key[1]} | {len(markets[key])} | {basis[key]} |")
    lines += ["", "## Split x category", "",
              "| split | category | markets | trades | win rate | mean price |", "|---|---|---|---|---|---|"]
    warnings = []
    for cell in table:
        win = "sealed" if cell["split"] == "test" else (
            "" if cell["win_rate"] is None else f"{cell['win_rate']:.3f}"
        )
        lines.append(
            f"| {cell['split']} | {cell['category']} | {cell['markets']} | {cell['trades']} | "
            f"{win} | {cell['mean_price']:.3f} |"
        )
        if cell["markets"] < MIN_MARKETS_PER_CELL:
            warnings.append(f"- {cell['split']} / {cell['category']}: only {cell['markets']} markets")
    lines += ["", f"## Small cells (fewer than {MIN_MARKETS_PER_CELL} markets)", ""]
    lines += warnings or ["- none"]
    lines += ["", "## Data quality", ""]
    if total:
        truncated = sum(row["history_truncated"] == 1 for row in features)
        lines.append(f"- Truncated wallet histories: {truncated} ({truncated / total:.1%})")
        bursts = sum(row["burst_size"] > 1 for row in features)
        lines.append(f"- Trades in multi-trade bursts: {bursts} ({bursts / total:.1%})")
        for name in OPTIONAL_FEATURES:
            missing = sum(row[f"missing_{name}"] for row in features)
            if missing:
                lines.append(f"- Missing `{name}`: {missing} ({missing / total:.1%})")
        for _, suffix in HORIZONS:
            statuses = Counter(
                row[f"label_status_{suffix}"] for row in labels if row["split"] == "train"
            )
            text = ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
            lines.append(f"- Label status +{suffix} (train only): {text}")
    lines += ["", "## Notes", "",
              "- `labels_test.csv` is sealed: do not open it until the final evaluation.",
              "- `overlap_random` markets traded before the cutoff and resolved after it. "
              "Report test results with and without them to check for leakage.",
              "- Group by `condition_id` (one selected market per event) for uncertainty estimates.",
              "- P&L is gross per share; fees are unknown, so net P&L is not provided."]
    return "\n".join(lines) + "\n"


def build_dataset(config: DatasetConfig) -> dict[str, Any]:
    if config.output.exists() and any(config.output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {config.output}")
    connection = open_readonly(config.db_path)
    try:
        rows, dropped = load_rows(connection, config.cohort, config.categories)
        summaries = load_wallet_summaries(connection, rows)
        stored_labels = load_labels(connection, (row["trade_key"] for row in rows))
    finally:
        connection.close()

    splits = assign_splits(rows, config.test_start, config.seed)
    features = build_features(rows, splits, summaries)
    labels = build_labels(rows, splits, stored_labels)
    labels_by_key = {row["trade_key"]: row for row in labels}

    config.output.mkdir(parents=True, exist_ok=True)
    files = {
        "features.csv": (FEATURE_COLUMNS, features),
        "labels_train.csv": (LABEL_COLUMNS, [row for row in labels if row["split"] == "train"]),
        "labels_test.csv": (LABEL_COLUMNS, [row for row in labels if row["split"] == "test"]),
    }
    for name, (columns, data) in files.items():
        _write_csv(config.output / name, columns, data)
    table = _summary_table(features, labels_by_key)
    (config.output / "report.md").write_text(
        _report(config, dropped, features, table, labels), encoding="utf-8",
    )
    counts = Counter(row["split"] for row in features)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database": str(config.db_path.resolve()),
        "cohort": config.cohort,
        "categories": list(config.categories),
        "test_start": config.test_start,
        "seed": config.seed,
        "feature_version": FEATURE_VERSION,
        "score_version": SCORE_VERSION,
        "cv_folds": CV_FOLDS,
        "burst_gap_seconds": BURST_GAP_SECONDS,
        "small_trade_inclusion": SMALL_TRADE_INCLUSION,
        "rows": {"total": len(features), "train": counts["train"], "test": counts["test"]},
        "dropped": dict(sorted(dropped.items())),
        "split_by_category": table,
        "excluded_fields": EXCLUDED_FIELDS,
        "files": {name: _sha256(config.output / name) for name in (*files, "report.md")},
    }
    (config.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest
