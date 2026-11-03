"""Cross-validated logistic-regression baseline for informed-flow trades.

Reads a dataset written by ``build-dataset`` and uses ONLY the training split:
``features.csv`` rows with ``split == "train"`` joined to ``labels_train.csv``.
``labels_test.csv`` is never opened here; the sealed test evaluation is a later,
separate step.

The model predicts the probability that the purchased outcome wins. The market's
own odds enter as ``logit(price)``, so other features can only add value by moving
the prediction away from the trade price. Hyperparameters are chosen by grouped
cross-validation on the dataset's market-level ``cv_fold`` column, scored by
weighted log loss, using the one-standard-error rule.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

MODEL_VERSION = 1
PRICE_CLIP = 0.001
DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_L1_RATIOS = (0.0, 0.5)  # 0 = L2 ridge, 0.5 = elastic net
WEIGHTINGS = ("none", "sample", "market", "sample_x_market")
FILTERS = {"all": None, "price_lt_0.95": 0.95}
FEATURE_SETS = ("price_only", "price_category", "full", "full_interactions")
FEATURE_SET_COMPLEXITY = {name: rank for rank, name in enumerate(FEATURE_SETS)}
EDGE_THRESHOLD = 0.02
BOOTSTRAP_SAMPLES = 1000

# Skewed, non-negative (or signed) quantities get a signed log1p transform.
LOG_FEATURES = (
    "notional_usd", "days_since_market_start", "hours_to_scheduled_end",
    "wallet_observed_trade_count", "wallet_distinct_market_count",
    "wallet_distinct_outcome_count", "wallet_active_days", "wallet_mean_notional_usd",
    "wallet_median_notional_usd", "wallet_p90_notional_usd", "wallet_max_notional_usd",
    "wallet_stddev_notional_usd", "wallet_cheap_trade_count", "wallet_last_7d_trade_count",
    "wallet_last_7d_notional_usd", "wallet_seconds_since_last_trade", "wallet_buy_count",
    "wallet_sell_count", "size_vs_median", "same_asset_buys_1h", "same_asset_notional_1h_usd",
    "burst_size",
)
RAW_FEATURES = (
    "fraction_of_market_life_elapsed", "wallet_mean_price",
    "wallet_largest_market_volume_share", "wallet_buy_notional_share", "history_truncated",
)
# One flag per distinct missingness pattern (the wallet flags move together).
MISSING_FLAGS = (
    "missing_days_since_market_start", "missing_hours_to_scheduled_end",
    "missing_fraction_of_market_life_elapsed", "missing_size_vs_median",
    "missing_wallet_mean_notional_usd",
)
# Deliberately excluded from inputs: identifiers (trade_key, condition_id, event_slug,
# proxy_wallet, burst_id), split bookkeeping, heuristic_score/bucket (benchmark only),
# outcome_name (free text), duplicated size columns (log_notional, size_shares), and
# calendar/time-trend columns (trade_ts, hour_utc, day_of_week, months_since_dataset_start)
# because the test split lies entirely after the training period.


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path
    output: Path
    horizon: str = "15m"
    seed: int = 0
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    l1_ratios: tuple[float, ...] = DEFAULT_L1_RATIOS
    weightings: tuple[str, ...] = WEIGHTINGS
    filters: tuple[str, ...] = tuple(FILTERS)
    feature_sets: tuple[str, ...] = FEATURE_SETS
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    jobs: int = -1  # parallel CPU workers for the CV grid; -1 = all cores


@dataclass
class TrainRows:
    trade_keys: list[str]
    markets: list[str]
    categories: list[str]
    folds: Any
    price: Any
    won: Any
    sample_weight: Any
    market_weight: Any
    heuristic: Any
    entry_price: Any  # delayed entry price at the chosen horizon (nan when unavailable)
    gross_pnl: Any  # gross per-share P&L at the chosen horizon (nan when unavailable)
    raw: dict[str, Any] = field(default_factory=dict)  # numeric feature columns (nan = missing)

    def subset(self, mask: Any) -> "TrainRows":
        import numpy as np

        index = np.flatnonzero(mask)
        pick = lambda values: [values[i] for i in index]  # noqa: E731
        return TrainRows(
            trade_keys=pick(self.trade_keys), markets=pick(self.markets),
            categories=pick(self.categories), folds=self.folds[index], price=self.price[index],
            won=self.won[index], sample_weight=self.sample_weight[index],
            market_weight=self.market_weight[index], heuristic=self.heuristic[index],
            entry_price=self.entry_price[index], gross_pnl=self.gross_pnl[index],
            raw={name: column[index] for name, column in self.raw.items()},
        )

    def __len__(self) -> int:
        return len(self.trade_keys)


def _float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_rows(dataset: Path, horizon: str) -> TrainRows:
    """Join training features to training labels. Never touches labels_test.csv."""
    import numpy as np

    features_path = dataset / "features.csv"
    labels_path = dataset / "labels_train.csv"
    for path in (features_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    labels: dict[str, dict[str, str]] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train":
                raise ValueError("labels_train.csv contains a non-train row")
            labels[row["trade_key"]] = row
    pnl_column = f"gross_pnl_per_share_{horizon}"
    entry_column = f"entry_price_{horizon}"
    status_column = f"label_status_{horizon}"
    sample = next(iter(labels.values()), None)
    if sample is not None and pnl_column not in sample:
        raise ValueError(f"Unknown horizon {horizon!r}; expected one of 15m, 1h, 24h")

    columns = (*LOG_FEATURES, *RAW_FEATURES, *MISSING_FLAGS)
    data: dict[str, list[Any]] = {name: [] for name in (
        "trade_key", "condition_id", "canonical_category", "cv_fold", "price", "won",
        "sample_weight", "market_weight", "heuristic_score", "entry", "pnl", *columns,
    )}
    with features_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                continue
            label = labels.get(row["trade_key"])
            if label is None:
                raise ValueError(f"Training trade has no label row: {row['trade_key']}")
            if label["outcome_won"] not in ("0", "1"):
                raise ValueError(f"Training trade has no outcome: {row['trade_key']}")
            complete = label[status_column] == "complete"
            data["trade_key"].append(row["trade_key"])
            data["condition_id"].append(row["condition_id"])
            data["canonical_category"].append(row["canonical_category"])
            data["cv_fold"].append(int(row["cv_fold"]))
            data["price"].append(float(row["price"]))
            data["won"].append(int(label["outcome_won"]))
            data["sample_weight"].append(float(row["sample_weight"]))
            data["market_weight"].append(float(row["market_weight"]))
            data["heuristic_score"].append(_float(row["heuristic_score"]))
            data["entry"].append(_float(label[entry_column]) if complete else math.nan)
            data["pnl"].append(_float(label[pnl_column]) if complete else math.nan)
            for name in columns:
                data[name].append(_float(row[name]))
    if not data["trade_key"]:
        raise ValueError("Dataset has no training rows")
    if len(data["trade_key"]) != len(labels):
        raise ValueError("Training features and labels_train.csv row counts differ")

    array = lambda name, dtype=float: np.asarray(data[name], dtype=dtype)  # noqa: E731
    return TrainRows(
        trade_keys=data["trade_key"], markets=data["condition_id"],
        categories=data["canonical_category"], folds=array("cv_fold", int),
        price=array("price"), won=array("won", int), sample_weight=array("sample_weight"),
        market_weight=array("market_weight"), heuristic=array("heuristic_score"),
        entry_price=array("entry"), gross_pnl=array("pnl"),
        raw={name: array(name) for name in columns},
    )


def logit(p: Any) -> Any:
    import numpy as np

    p = np.clip(p, PRICE_CLIP, 1 - PRICE_CLIP)
    return np.log(p / (1 - p))


def signed_log1p(x: Any) -> Any:
    import numpy as np

    return np.sign(x) * np.log1p(np.abs(x))


class FeatureBuilder:
    """Turns raw columns into a model matrix. Medians and scaling come from fit()."""

    def __init__(self, feature_set: str, categories: Sequence[str]) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}")
        self.feature_set = feature_set
        self.categories = sorted(set(categories))
        self.medians: dict[str, float] = {}
        self.means: Any = None
        self.scales: Any = None
        self.names: list[str] = []

    def _numeric(self, rows: TrainRows) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        for name in LOG_FEATURES:
            columns[f"log_{name}"] = signed_log1p(rows.raw[name])
        for name in RAW_FEATURES:
            columns[name] = rows.raw[name]
        return columns

    def _matrix(self, rows: TrainRows) -> tuple[list[str], Any]:
        import numpy as np

        names = ["logit_price"]
        columns = [logit(rows.price)]
        if self.feature_set != "price_only":
            for category in self.categories:
                names.append(f"category={category}")
                columns.append(np.asarray([c == category for c in rows.categories], dtype=float))
        if self.feature_set in ("full", "full_interactions"):
            numeric = self._numeric(rows)
            for name, values in numeric.items():
                filled = np.where(np.isnan(values), self.medians[name], values)
                names.append(name)
                columns.append(filled)
            for flag in MISSING_FLAGS:
                names.append(flag)
                columns.append(np.nan_to_num(rows.raw[flag], nan=1.0))
            if self.feature_set == "full_interactions":
                base = logit(rows.price)
                for name, values in numeric.items():
                    filled = np.where(np.isnan(values), self.medians[name], values)
                    names.append(f"{name}*logit_price")
                    columns.append(filled * base)
        return names, np.column_stack(columns)

    def fit(self, rows: TrainRows) -> "FeatureBuilder":
        import numpy as np

        if self.feature_set in ("full", "full_interactions"):
            for name, values in self._numeric(rows).items():
                finite = values[~np.isnan(values)]
                self.medians[name] = float(np.median(finite)) if finite.size else 0.0
        self.names, matrix = self._matrix(rows)
        self.means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        self.scales = np.where(scales > 0, scales, 1.0)
        return self

    def transform(self, rows: TrainRows) -> Any:
        _, matrix = self._matrix(rows)
        return (matrix - self.means) / self.scales


def training_weights(rows: TrainRows, weighting: str) -> Any:
    import numpy as np

    if weighting == "none":
        weights = np.ones(len(rows))
    elif weighting == "sample":
        weights = rows.sample_weight.copy()
    elif weighting == "market":
        weights = rows.market_weight.copy()
    elif weighting == "sample_x_market":
        weights = rows.sample_weight * rows.market_weight
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    return weights * (len(weights) / weights.sum())


def evaluation_weights(rows: TrainRows) -> dict[str, Any]:
    """Fixed evaluation weights, identical for every candidate so scores compare."""
    return {
        # Each market counts about once; sampled small trades are re-inflated.
        "market": rows.sample_weight * rows.market_weight,
        # Each trade counts once, corrected for the 25% small-trade sample.
        "trade": rows.sample_weight.copy(),
    }


def _metrics(y: Any, p: Any, weights: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = np.clip(p, 1e-6, 1 - 1e-6)
    result = {
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
    }
    result["auc"] = (float(roc_auc_score(y, p, sample_weight=weights))
                     if len(set(y.tolist())) == 2 else math.nan)
    return result


@dataclass(frozen=True)
class Candidate:
    filter_name: str
    feature_set: str
    weighting: str
    c: float  # math.inf means unpenalized
    l1_ratio: float

    @property
    def penalty(self) -> str:
        if math.isinf(self.c):
            return "none"
        return "l2" if self.l1_ratio == 0 else f"elasticnet({self.l1_ratio:g})"

    @property
    def key(self) -> str:
        c = "inf" if math.isinf(self.c) else f"{self.c:g}"
        return f"{self.filter_name}|{self.feature_set}|{self.weighting}|{self.penalty}|C={c}"

    def simplicity(self) -> tuple[float, ...]:
        # Lower sorts as simpler: fewer features, L2 before elastic net, stronger penalty.
        c_rank = -1.0 if math.isinf(self.c) else self.c
        return (FEATURE_SET_COMPLEXITY[self.feature_set], self.l1_ratio, c_rank)


def candidates_for(config: TrainConfig, filter_name: str) -> list[Candidate]:
    result: list[Candidate] = []
    for weighting in config.weightings:
        for feature_set in config.feature_sets:
            if feature_set == "price_only":
                result.append(Candidate(filter_name, feature_set, weighting, math.inf, 0.0))
                continue
            for l1_ratio in config.l1_ratios:
                if feature_set == "price_category" and l1_ratio != 0:
                    continue  # elastic net is pointless with a handful of dummies
                for c in config.c_grid:
                    result.append(Candidate(filter_name, feature_set, weighting, c, l1_ratio))
    return result


def _model(candidate: Candidate, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    if math.isinf(candidate.c):
        return LogisticRegression(C=math.inf, max_iter=5000)
    if candidate.l1_ratio == 0:
        return LogisticRegression(C=candidate.c, l1_ratio=0.0, max_iter=5000)
    return LogisticRegression(C=candidate.c, l1_ratio=candidate.l1_ratio, solver="saga",
                              max_iter=20000, tol=1e-4, random_state=seed)


def fit_candidate(candidate: Candidate, rows: TrainRows, categories: Sequence[str],
                  seed: int) -> tuple[FeatureBuilder, Any]:
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    builder = FeatureBuilder(candidate.feature_set, categories).fit(rows)
    model = _model(candidate, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(builder.transform(rows), rows.won,
                  sample_weight=training_weights(rows, candidate.weighting))
    return builder, model


def out_of_fold(candidate: Candidate, rows: TrainRows, categories: Sequence[str],
                seed: int) -> Any:
    import numpy as np

    predictions = np.full(len(rows), np.nan)
    for fold in sorted(set(rows.folds.tolist())):
        train_mask = rows.folds != fold
        builder, model = fit_candidate(candidate, rows.subset(train_mask), categories, seed)
        test_rows = rows.subset(~train_mask)
        predictions[~train_mask] = model.predict_proba(builder.transform(test_rows))[:, 1]
    return predictions


def fold_scores(rows: TrainRows, predictions: Any) -> dict[str, Any]:
    """Per-fold and pooled metrics under both evaluation weightings."""
    import numpy as np

    weights = evaluation_weights(rows)
    per_fold = []
    for fold in sorted(set(rows.folds.tolist())):
        mask = rows.folds == fold
        per_fold.append(_metrics(rows.won[mask], predictions[mask], weights["market"][mask])["log_loss"])
    per_fold_arr = np.asarray(per_fold)
    pooled_market = _metrics(rows.won, predictions, weights["market"])
    pooled_trade = _metrics(rows.won, predictions, weights["trade"])
    return {
        "fold_log_loss": per_fold,
        "mean_log_loss": float(per_fold_arr.mean()),
        "se_log_loss": float(per_fold_arr.std(ddof=1) / math.sqrt(len(per_fold_arr)))
        if len(per_fold_arr) > 1 else 0.0,
        "market_weighted": pooled_market,
        "trade_weighted": pooled_trade,
    }


def _score_candidate(index: int, candidate: Candidate, rows: TrainRows,
                     categories: Sequence[str], seed: int) -> tuple[int, Any, dict[str, Any], float]:
    started = time.perf_counter()
    predictions = out_of_fold(candidate, rows, categories, seed)
    return index, predictions, fold_scores(rows, predictions), time.perf_counter() - started


def _expected_cost(candidate: Candidate) -> float:
    """Rough relative fit time, used to start the slow elastic-net fits first."""
    if math.isinf(candidate.c):
        return 0.0
    cost = 1.0 + FEATURE_SET_COMPLEXITY[candidate.feature_set]
    if candidate.l1_ratio > 0:
        cost *= 10 * (1 + candidate.c)
    return cost


def _setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"informed_flow.train_baseline.{output.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(output / "train.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def select_one_se(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (best_by_mean, chosen_by_one_standard_error_rule)."""
    best = min(results, key=lambda r: r["mean_log_loss"])
    threshold = best["mean_log_loss"] + best["se_log_loss"]
    eligible = [r for r in results if r["mean_log_loss"] <= threshold]
    chosen = min(eligible, key=lambda r: (r["candidate"].simplicity(), r["mean_log_loss"]))
    return best, chosen


def _weighted_mean(values: Any, weights: Any) -> float:
    import numpy as np

    mask = ~np.isnan(values)
    if not mask.any() or weights[mask].sum() == 0:
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def follow_diagnostic(rows: TrainRows, predictions: Any, seed: int,
                      samples: int) -> dict[str, Any]:
    """Gross P&L of trades where the prediction beats the delayed entry price.

    This is an out-of-fold, in-training-period diagnostic only: it is not a
    profitability claim and not a trading rule.
    """
    import numpy as np

    available = ~np.isnan(rows.entry_price) & ~np.isnan(rows.gross_pnl)
    flagged = available & (predictions - rows.entry_price > EDGE_THRESHOLD)
    weights = evaluation_weights(rows)

    def group(mask: Any) -> dict[str, Any]:
        markets = sorted({rows.markets[i] for i in np.flatnonzero(mask)})
        summary = {
            "trades": int(mask.sum()),
            "markets": len(markets),
            "mean_gross_pnl_trade_weighted": _weighted_mean(
                np.where(mask, rows.gross_pnl, np.nan), weights["trade"]),
            "mean_gross_pnl_market_weighted": _weighted_mean(
                np.where(mask, rows.gross_pnl, np.nan), weights["market"]),
            "ci95_trade_weighted": None,
        }
        if len(markets) >= 2 and samples > 0:
            rng = np.random.default_rng(seed)
            by_market: dict[str, list[int]] = {}
            for i in np.flatnonzero(mask):
                by_market.setdefault(rows.markets[i], []).append(i)
            groups = list(by_market.values())
            estimates = []
            for _ in range(samples):
                picks = rng.integers(0, len(groups), len(groups))
                index = np.concatenate([groups[j] for j in picks])
                estimates.append(float(np.average(rows.gross_pnl[index],
                                                  weights=weights["trade"][index])))
            summary["ci95_trade_weighted"] = [float(np.percentile(estimates, 2.5)),
                                              float(np.percentile(estimates, 97.5))]
        return summary

    return {
        "edge_threshold": EDGE_THRESHOLD,
        "rows_with_delayed_price": int(available.sum()),
        "flagged": group(flagged),
        "all_with_delayed_price": group(available),
    }


def calibration_table(rows: TrainRows, predictions: Any, bins: int = 10) -> list[dict[str, Any]]:
    import numpy as np

    weights = evaluation_weights(rows)["trade"]
    edges = np.quantile(predictions, np.linspace(0, 1, bins + 1))
    table = []
    for i in range(bins):
        upper = predictions <= edges[i + 1] if i == bins - 1 else predictions < edges[i + 1]
        mask = (predictions >= edges[i]) & upper
        if not mask.any():
            continue
        table.append({
            "bin": i + 1, "trades": int(mask.sum()),
            "mean_predicted": float(np.average(predictions[mask], weights=weights[mask])),
            "observed_win_rate": float(np.average(rows.won[mask], weights=weights[mask])),
            "mean_trade_price": float(np.average(rows.price[mask], weights=weights[mask])),
        })
    return table


def by_category(rows: TrainRows, predictions: Any, baseline: Any) -> list[dict[str, Any]]:
    import numpy as np

    weights = evaluation_weights(rows)["market"]
    result = []
    for category in sorted(set(rows.categories)):
        mask = np.asarray([c == category for c in rows.categories])
        model = _metrics(rows.won[mask], predictions[mask], weights[mask])
        base = _metrics(rows.won[mask], baseline[mask], weights[mask])
        result.append({
            "category": category, "trades": int(mask.sum()),
            "markets": len({rows.markets[i] for i in np.flatnonzero(mask)}),
            "model_log_loss": model["log_loss"], "price_log_loss": base["log_loss"],
            "model_auc": model["auc"], "price_auc": base["auc"],
        })
    return result


def baselines(rows: TrainRows, categories: Sequence[str], seed: int) -> dict[str, Any]:
    """Reference predictions on the same folds as the candidates."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    weights = evaluation_weights(rows)
    constant = np.full(len(rows), np.nan)
    heuristic = np.full(len(rows), np.nan)
    for fold in sorted(set(rows.folds.tolist())):
        train = rows.folds != fold
        constant[~train] = np.average(rows.won[train], weights=weights["trade"][train])
        # Heuristic benchmark: does the hand-written score add anything to price?
        x = np.column_stack([logit(rows.price), np.nan_to_num(rows.heuristic)])
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        model = LogisticRegression(C=math.inf, max_iter=5000)
        model.fit((x[train] - mean) / std, rows.won[train])
        heuristic[~train] = model.predict_proba((x[~train] - mean) / std)[:, 1]
    raw_price = np.clip(rows.price, PRICE_CLIP, 1 - PRICE_CLIP)
    result = {
        "constant": fold_scores(rows, constant),
        "raw_trade_price": fold_scores(rows, raw_price),
        "price_plus_heuristic": fold_scores(rows, heuristic),
    }
    delayed = ~np.isnan(rows.entry_price)
    if delayed.any():
        result["delayed_price_subset"] = {
            "rows": int(delayed.sum()),
            "raw_trade_price": _metrics(rows.won[delayed], raw_price[delayed], weights["market"][delayed]),
            "delayed_entry_price": _metrics(
                rows.won[delayed], np.clip(rows.entry_price[delayed], PRICE_CLIP, 1 - PRICE_CLIP),
                weights["market"][delayed]),
        }
    return result


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def train_baseline(config: TrainConfig, progress: Any = None) -> dict[str, Any]:
    """Run the CV grid, pick a model per trade filter, refit, and write outputs."""
    if config.output.exists():
        raise FileExistsError(f"Output folder already exists; choose a new name: {config.output}")
    unknown = [f for f in config.filters if f not in FILTERS]
    if unknown:
        raise ValueError(f"Unknown filters: {unknown}")
    all_rows = load_training_rows(config.dataset, config.horizon)
    config.output.mkdir(parents=True)
    logger = _setup_logger(config.output)

    def say(message: str) -> None:
        logger.info(message)
        if progress is not None:
            progress(message)

    try:
        return _run(config, all_rows, say)
    except Exception:
        logger.exception("Training failed; this output folder is incomplete. "
                         "Rerun with a new --output name.")
        raise
    finally:
        _close_logger(logger)


def _run(config: TrainConfig, all_rows: TrainRows, say: Any) -> dict[str, Any]:
    import numpy as np
    import sklearn
    from joblib import Parallel, delayed

    categories = sorted(set(all_rows.categories))
    run_started = time.perf_counter()
    say(f"Training rows: {len(all_rows)} trades, {len(set(all_rows.markets))} markets; "
        f"horizon +{config.horizon}; seed {config.seed}; parallel jobs {config.jobs}")

    experiments: dict[str, Any] = {}
    cv_records: list[dict[str, Any]] = []
    oof_columns: dict[str, Any] = {}
    for filter_name in config.filters:
        limit = FILTERS[filter_name]
        rows = all_rows if limit is None else all_rows.subset(all_rows.price < limit)
        if len(set(rows.folds.tolist())) < 2:
            raise ValueError(f"Filter {filter_name} leaves fewer than two CV folds")
        candidates = candidates_for(config, filter_name)
        say(f"{filter_name}: {len(rows)} trades, {len(set(rows.markets))} markets, "
            f"{len(candidates)} candidates x {len(set(rows.folds.tolist()))} folds")
        order = sorted(range(len(candidates)), key=lambda i: -_expected_cost(candidates[i]))
        slots: list[dict[str, Any] | None] = [None] * len(candidates)
        filter_started = time.perf_counter()
        tasks = (delayed(_score_candidate)(i, candidates[i], rows, categories, config.seed)
                 for i in order)
        done = 0
        for index, predictions, scores, seconds in Parallel(
                n_jobs=config.jobs, return_as="generator_unordered")(tasks):
            slots[index] = {"candidate": candidates[index], "predictions": predictions, **scores}
            done += 1
            if seconds > 30:
                say(f"  slow candidate {candidates[index].key}: {seconds:.0f}s")
            if done % 10 == 0 or done == len(candidates):
                elapsed = time.perf_counter() - filter_started
                remaining = elapsed / done * (len(candidates) - done)
                say(f"  {done}/{len(candidates)} candidates scored; elapsed {elapsed / 60:.1f} min; "
                    f"about {remaining / 60:.1f} min left for this filter (slowest fits run first)")
        results = [slot for slot in slots if slot is not None]

        price_reference = {r["candidate"].weighting: r for r in results
                           if r["candidate"].feature_set == "price_only"}
        for r in results:
            reference = price_reference[r["candidate"].weighting]
            diffs = np.asarray(r["fold_log_loss"]) - np.asarray(reference["fold_log_loss"])
            r["delta_vs_price_only"] = float(diffs.mean())
            r["delta_se"] = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0

        best, chosen = select_one_se(results)
        say(f"{filter_name}: chosen {chosen['candidate'].key} "
            f"(mean log loss {chosen['mean_log_loss']:.4f} +/- {chosen['se_log_loss']:.4f}; "
            f"vs price-only {chosen['delta_vs_price_only']:+.4f} +/- {chosen['delta_se']:.4f}); "
            f"best by mean {best['candidate'].key}")
        say(f"{filter_name}: computing baselines, diagnostics, and the final refit")
        price_only = min((r for r in results if r["candidate"].feature_set == "price_only"),
                         key=lambda r: r["mean_log_loss"])
        base = baselines(rows, categories, config.seed)
        builder, model = fit_candidate(chosen["candidate"], rows, categories, config.seed)
        coefficients = sorted(
            ({"feature": name, "coefficient_per_sd": float(coef)}
             for name, coef in zip(builder.names, model.coef_[0])),
            key=lambda item: -abs(item["coefficient_per_sd"]),
        )
        experiments[filter_name] = {
            "rows": len(rows), "markets": len(set(rows.markets)),
            "wallets_unknown": "wallet ids are not model inputs",
            "candidates": len(candidates),
            "best_by_mean": best["candidate"].key,
            "chosen": chosen["candidate"].key,
            "chosen_scores": {k: chosen[k] for k in (
                "mean_log_loss", "se_log_loss", "market_weighted", "trade_weighted",
                "delta_vs_price_only", "delta_se")},
            "best_scores": {k: best[k] for k in (
                "mean_log_loss", "se_log_loss", "delta_vs_price_only", "delta_se")},
            "price_only_scores": {k: price_only[k] for k in (
                "mean_log_loss", "se_log_loss", "market_weighted", "trade_weighted")},
            "baselines": base,
            "by_category": by_category(rows, chosen["predictions"], price_only["predictions"]),
            "calibration": calibration_table(rows, chosen["predictions"]),
            "follow_diagnostic": {
                "chosen_model": follow_diagnostic(rows, chosen["predictions"], config.seed,
                                                  config.bootstrap_samples),
                "price_only_model": follow_diagnostic(rows, price_only["predictions"], config.seed,
                                                      config.bootstrap_samples),
            },
            "final_model": {
                "candidate": chosen["candidate"].key,
                "feature_set": chosen["candidate"].feature_set,
                "weighting": chosen["candidate"].weighting,
                "C": chosen["candidate"].c, "l1_ratio": chosen["candidate"].l1_ratio,
                "features": builder.names,
                "categories": builder.categories,
                "medians": builder.medians,
                "means": builder.means.tolist(), "scales": builder.scales.tolist(),
                "intercept": float(model.intercept_[0]),
                "coefficients": [float(c) for c in model.coef_[0]],
                "price_clip": PRICE_CLIP,
                "trained_rows": len(rows),
            },
            "coefficients": coefficients,
        }
        for r in results:
            c = r["candidate"]
            cv_records.append({
                "filter": filter_name, "feature_set": c.feature_set, "weighting": c.weighting,
                "penalty": c.penalty, "C": "inf" if math.isinf(c.c) else c.c,
                "l1_ratio": c.l1_ratio, "mean_log_loss": r["mean_log_loss"],
                "se_log_loss": r["se_log_loss"],
                "delta_vs_price_only": r["delta_vs_price_only"], "delta_se": r["delta_se"],
                "pooled_log_loss_market": r["market_weighted"]["log_loss"],
                "pooled_brier_market": r["market_weighted"]["brier"],
                "pooled_auc_market": r["market_weighted"]["auc"],
                "pooled_log_loss_trade": r["trade_weighted"]["log_loss"],
                "pooled_auc_trade": r["trade_weighted"]["auc"],
                "chosen": r is chosen, "best_by_mean": r is best,
                **{f"fold_{i}_log_loss": v for i, v in enumerate(r["fold_log_loss"])},
            })
        oof_columns[filter_name] = (rows, chosen["predictions"], price_only["predictions"])

    _write_cv(config.output / "cv_results.csv", cv_records)
    _write_oof(config.output / "oof_predictions.csv", oof_columns)
    for filter_name, experiment in experiments.items():
        (config.output / f"model_{filter_name}.json").write_text(
            json.dumps(_clean(experiment["final_model"]), indent=2) + "\n", encoding="utf-8")
        with (config.output / f"coefficients_{filter_name}.csv").open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=["feature", "coefficient_per_sd"])
            writer.writeheader()
            writer.writerows(experiment["coefficients"])
    dataset_manifest = config.dataset / "manifest.json"
    manifest = {
        "model_version": MODEL_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(config.dataset.resolve()),
        "dataset_manifest_sha256": _sha256(dataset_manifest) if dataset_manifest.is_file() else None,
        "inputs_sha256": {name: _sha256(config.dataset / name)
                          for name in ("features.csv", "labels_train.csv")},
        "labels_test_opened": False,
        "horizon": config.horizon, "seed": config.seed,
        "grid": {"C": list(config.c_grid), "l1_ratio": list(config.l1_ratios),
                 "weightings": list(config.weightings), "filters": list(config.filters),
                 "feature_sets": list(config.feature_sets)},
        "selection": "minimum mean fold log loss (sample_weight x market_weight); "
                     "simplest candidate within one standard error",
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "parallel_jobs": config.jobs, "status": "complete",
        "experiments": experiments,
    }
    (config.output / "manifest.json").write_text(
        json.dumps(_clean(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (config.output / "report.md").write_text(render_report(manifest), encoding="utf-8")
    say(f"Finished in {(time.perf_counter() - run_started) / 60:.1f} min; wrote {config.output}")
    return manifest


def _write_cv(path: Path, records: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_oof(path: Path, columns: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filter", "trade_key", "condition_id", "canonical_category", "cv_fold",
                         "price", "outcome_won", "p_chosen", "p_price_only"])
        for filter_name, (rows, chosen, price_only) in columns.items():
            for i in range(len(rows)):
                writer.writerow([filter_name, rows.trade_keys[i], rows.markets[i], rows.categories[i],
                                 int(rows.folds[i]), rows.price[i], int(rows.won[i]),
                                 f"{chosen[i]:.6f}", f"{price_only[i]:.6f}"])


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Logistic-regression baseline (model version {manifest['model_version']})", "",
        f"- Dataset: `{manifest['dataset']}` (training split only; `labels_test.csv` not opened)",
        f"- Follow horizon for P&L diagnostics: +{manifest['horizon']}; seed {manifest['seed']}",
        f"- Selection: {manifest['selection']}",
        "- All numbers below are out-of-fold results inside the training period. They are "
        "model-selection diagnostics, not a test result and not a profitability claim.", "",
    ]
    for filter_name, exp in manifest["experiments"].items():
        chosen = exp["chosen_scores"]
        price = exp["price_only_scores"]
        base = exp["baselines"]
        lines += [
            f"## Trade filter: `{filter_name}`", "",
            f"{exp['rows']} trades in {exp['markets']} markets; {exp['candidates']} candidates.", "",
            f"- **Chosen (1-SE rule):** `{exp['chosen']}`",
            f"- Best by mean log loss: `{exp['best_by_mean']}` "
            f"(delta vs price-only {_fmt(exp['best_scores']['delta_vs_price_only'])} "
            f"+/- {_fmt(exp['best_scores']['delta_se'])})", "",
            "| Predictor | Fold-mean log loss (market-wt) | Pooled AUC (market-wt) | Pooled AUC (trade-wt) |",
            "|---|---|---|---|",
            f"| Constant win rate | {_fmt(base['constant']['mean_log_loss'])} | "
            f"{_fmt(base['constant']['market_weighted']['auc'])} | {_fmt(base['constant']['trade_weighted']['auc'])} |",
            f"| Raw trade price | {_fmt(base['raw_trade_price']['mean_log_loss'])} | "
            f"{_fmt(base['raw_trade_price']['market_weighted']['auc'])} | "
            f"{_fmt(base['raw_trade_price']['trade_weighted']['auc'])} |",
            f"| Price-only logistic | {_fmt(price['mean_log_loss'])} | "
            f"{_fmt(price['market_weighted']['auc'])} | {_fmt(price['trade_weighted']['auc'])} |",
            f"| Price + heuristic score | {_fmt(base['price_plus_heuristic']['mean_log_loss'])} | "
            f"{_fmt(base['price_plus_heuristic']['market_weighted']['auc'])} | "
            f"{_fmt(base['price_plus_heuristic']['trade_weighted']['auc'])} |",
            f"| **Chosen model** | {_fmt(chosen['mean_log_loss'])} | "
            f"{_fmt(chosen['market_weighted']['auc'])} | {_fmt(chosen['trade_weighted']['auc'])} |", "",
            f"Chosen minus price-only log loss: {_fmt(chosen['delta_vs_price_only'])} "
            f"+/- {_fmt(chosen['delta_se'])} (negative = better).", "",
        ]
        delayed = base.get("delayed_price_subset")
        if delayed:
            lines += [
                f"On the {delayed['rows']} rows with a +{manifest['horizon']} price, the delayed "
                f"price alone scores log loss {_fmt(delayed['delayed_entry_price']['log_loss'])} vs "
                f"{_fmt(delayed['raw_trade_price']['log_loss'])} for the trade price (the market "
                "keeps learning after the trade).", "",
            ]
        lines += ["### By category (market-weighted)", "",
                  "| Category | Markets | Trades | Model log loss | Price-only log loss | Model AUC | Price AUC |",
                  "|---|---|---|---|---|---|---|"]
        for item in exp["by_category"]:
            lines.append(
                f"| {item['category']} | {item['markets']} | {item['trades']} | "
                f"{_fmt(item['model_log_loss'])} | {_fmt(item['price_log_loss'])} | "
                f"{_fmt(item['model_auc'])} | {_fmt(item['price_auc'])} |")
        lines += ["", "### Calibration (out-of-fold, trade-weighted deciles)", "",
                  "| Bin | Trades | Mean predicted | Observed win rate | Mean trade price |",
                  "|---|---|---|---|---|"]
        for item in exp["calibration"]:
            lines.append(f"| {item['bin']} | {item['trades']} | {_fmt(item['mean_predicted'], 3)} | "
                         f"{_fmt(item['observed_win_rate'], 3)} | {_fmt(item['mean_trade_price'], 3)} |")
        lines += ["", f"### Follow diagnostic at +{manifest['horizon']} "
                      f"(prediction - delayed price > {EDGE_THRESHOLD})", "",
                  "| Predictions from | Group | Trades | Markets | Gross P&L/share (trade-wt) | "
                  "95% CI (market bootstrap) | Gross P&L/share (market-wt) |",
                  "|---|---|---|---|---|---|---|"]
        for source, diag in exp["follow_diagnostic"].items():
            for group_name in ("flagged", "all_with_delayed_price"):
                g = diag[group_name]
                ci = g["ci95_trade_weighted"]
                ci_text = f"{_fmt(ci[0])} to {_fmt(ci[1])}" if ci else "n/a"
                lines.append(
                    f"| {source} | {group_name} | {g['trades']} | {g['markets']} | "
                    f"{_fmt(g['mean_gross_pnl_trade_weighted'])} | {ci_text} | "
                    f"{_fmt(g['mean_gross_pnl_market_weighted'])} |")
        lines += ["", "### Largest coefficients (per standard deviation)", "",
                  "| Feature | Coefficient |", "|---|---|"]
        for item in exp["coefficients"][:15]:
            lines.append(f"| `{item['feature']}` | {_fmt(item['coefficient_per_sd'])} |")
        lines.append("")
    lines += [
        "## Notes", "",
        "- Gross P&L ignores fees and order-book depth, which are unknown historically.",
        "- Folds come from the dataset's market-level `cv_fold`, so no market is in both the "
        "fitting and scoring part of a fold. Folds are random, not time-ordered.",
        "- The sealed test set has not been used. Evaluate the chosen model on it once, "
        "with and without `overlap_random` markets.",
        "",
    ]
    return "\n".join(lines)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected a comma-separated list of numbers")
    return values


def parse_name_list(text: str, allowed: Iterable[str]) -> tuple[str, ...]:
    allowed = tuple(allowed)
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [v for v in values if v not in allowed]
    if not values or unknown:
        raise ValueError(f"expected names from {', '.join(allowed)}")
    return values
