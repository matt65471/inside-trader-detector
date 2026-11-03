from __future__ import annotations

import csv
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional model extras
    np = None

if np is not None:
    from informed_flow.baseline_model import (
        LOG_FEATURES, MISSING_FLAGS, RAW_FEATURES, Candidate, FeatureBuilder, TrainConfig,
        load_training_rows, select_one_se, train_baseline,
    )
    from informed_flow.cli import main

CATEGORIES = ("finance", "geopolitics", "politics", "sports")


def write_dataset(folder: Path, *, signal: bool, markets: int = 60, per_market: int = 12,
                  seed: int = 1) -> None:
    """Synthetic build-dataset output. With signal, large size_vs_median trades win more."""
    rng = random.Random(seed)
    folder.mkdir(parents=True)
    feature_fields = [
        "trade_key", "condition_id", "event_slug", "proxy_wallet", "canonical_category", "split",
        "split_basis", "cv_fold", "price", "heuristic_score", "market_weight", "sample_weight",
        *LOG_FEATURES, *RAW_FEATURES, *MISSING_FLAGS,
    ]
    label_fields = ["trade_key", "split", "outcome_won", "edge",
                    "entry_price_15m", "gross_pnl_per_share_15m", "label_status_15m"]
    features, labels = [], []
    for m in range(markets):
        category = CATEGORIES[m % len(CATEGORIES)]
        fold = m % 5
        for t in range(per_market):
            price = rng.uniform(0.05, 0.95)
            size_ratio = rng.lognormvariate(0, 1)
            p_win = price
            if signal:
                shift = 1.5 * (math.log(size_ratio))
                p_win = 1 / (1 + math.exp(-(math.log(price / (1 - price)) + shift)))
            won = int(rng.random() < p_win)
            entry = min(0.99, max(0.01, price + rng.uniform(-0.02, 0.02)))
            key = f"m{m}t{t}"
            row = {name: str(rng.uniform(0, 100)) for name in (*LOG_FEATURES, *RAW_FEATURES)}
            row["size_vs_median"] = str(size_ratio)
            row["history_truncated"] = "0"
            for flag in MISSING_FLAGS:
                row[flag] = "0"
            if t == 0:  # a missing value that must be median-filled
                row["wallet_mean_notional_usd"] = ""
                row["missing_wallet_mean_notional_usd"] = "1"
            row.update({
                "trade_key": key, "condition_id": f"cond{m}", "event_slug": f"event{m}",
                "proxy_wallet": f"w{rng.randrange(40)}", "canonical_category": category,
                "split": "train", "split_basis": "before_cutoff", "cv_fold": str(fold),
                "price": f"{price:.4f}", "heuristic_score": str(rng.randrange(10)),
                "market_weight": str(1 / per_market), "sample_weight": "1.0",
            })
            features.append(row)
            labels.append({
                "trade_key": key, "split": "train", "outcome_won": str(won),
                "edge": f"{won - price:.4f}", "entry_price_15m": f"{entry:.4f}",
                "gross_pnl_per_share_15m": f"{won - entry:.4f}", "label_status_15m": "complete",
            })
    # A test row whose label lives only in the sealed file; it must be ignored.
    test_row = dict(features[0], trade_key="sealed", split="test", cv_fold="")
    features.append(test_row)
    for path, fields, rows in ((folder / "features.csv", feature_fields, features),
                               (folder / "labels_train.csv", label_fields, labels)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")


@unittest.skipIf(np is None, "model extras (numpy, scikit-learn) not installed")
class BaselineModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def small_config(self, dataset: Path, output: Path, **overrides) -> TrainConfig:
        settings = dict(
            dataset=dataset, output=output, c_grid=(0.01, 1.0), l1_ratios=(0.0,),
            weightings=("none", "sample_x_market"), filters=("all",),
            feature_sets=("price_only", "price_category", "full"), bootstrap_samples=50, jobs=1,
        )
        settings.update(overrides)
        return TrainConfig(**settings)

    def test_loads_train_only_without_test_labels(self) -> None:
        dataset = self.root / "ds"
        write_dataset(dataset, signal=False, markets=10, per_market=3)
        rows = load_training_rows(dataset, "15m")
        self.assertEqual(len(rows), 30)
        self.assertNotIn("sealed", rows.trade_keys)
        self.assertFalse((dataset / "labels_test.csv").exists())

    def test_missing_label_is_an_error(self) -> None:
        dataset = self.root / "ds"
        write_dataset(dataset, signal=False, markets=5, per_market=2)
        lines = (dataset / "labels_train.csv").read_text(encoding="utf-8").splitlines()
        (dataset / "labels_train.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_training_rows(dataset, "15m")

    def test_feature_builder_uses_training_medians(self) -> None:
        dataset = self.root / "ds"
        write_dataset(dataset, signal=False, markets=10, per_market=4)
        rows = load_training_rows(dataset, "15m")
        train = rows.subset(rows.folds != 0)
        held_out = rows.subset(rows.folds == 0)
        builder = FeatureBuilder("full", CATEGORIES).fit(train)
        matrix = builder.transform(held_out)
        self.assertFalse(np.isnan(matrix).any())
        column = builder.names.index("log_wallet_mean_notional_usd")
        missing = np.isnan(held_out.raw["wallet_mean_notional_usd"])
        self.assertTrue(missing.any())
        expected = (builder.medians["log_wallet_mean_notional_usd"] - builder.means[column]) / builder.scales[column]
        np.testing.assert_allclose(matrix[missing, column], expected)

    def test_one_standard_error_rule_prefers_simpler_model(self) -> None:
        simple = Candidate("all", "price_only", "none", math.inf, 0.0)
        complex_ = Candidate("all", "full", "none", 1.0, 0.0)
        strong = Candidate("all", "full", "none", 0.01, 0.0)
        results = [
            {"candidate": complex_, "mean_log_loss": 0.500, "se_log_loss": 0.010},
            {"candidate": strong, "mean_log_loss": 0.505, "se_log_loss": 0.010},
            {"candidate": simple, "mean_log_loss": 0.530, "se_log_loss": 0.010},
        ]
        best, chosen = select_one_se(results)
        self.assertIs(best["candidate"], complex_)
        self.assertIs(chosen["candidate"], strong)

    def test_detects_signal_and_writes_outputs(self) -> None:
        dataset, output = self.root / "ds", self.root / "model"
        write_dataset(dataset, signal=True)
        manifest = train_baseline(self.small_config(dataset, output))
        experiment = manifest["experiments"]["all"]
        self.assertNotEqual(experiment["final_model"]["feature_set"], "price_only")
        self.assertLess(experiment["chosen_scores"]["delta_vs_price_only"], 0)
        self.assertEqual(experiment["coefficients"][0]["feature"], "log_size_vs_median")
        for name in ("manifest.json", "report.md", "train.log", "cv_results.csv", "oof_predictions.csv",
                     "model_all.json", "coefficients_all.csv"):
            self.assertTrue((output / name).is_file(), name)
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["labels_test_opened"])
        with (output / "cv_results.csv").open(encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(sum(r["chosen"] == "True" for r in records), 1)
        self.assertIn("not a profitability claim", (output / "report.md").read_text(encoding="utf-8"))
        log = (output / "train.log").read_text(encoding="utf-8")
        self.assertIn("candidates scored", log)
        self.assertIn("Finished in", log)

    def test_parallel_matches_serial(self) -> None:
        dataset = self.root / "ds"
        write_dataset(dataset, signal=True, markets=30, per_market=8)
        serial = train_baseline(self.small_config(dataset, self.root / "serial"))
        parallel = train_baseline(self.small_config(dataset, self.root / "parallel", jobs=2))
        for key in ("chosen", "best_by_mean"):
            self.assertEqual(serial["experiments"]["all"][key], parallel["experiments"]["all"][key])
        self.assertAlmostEqual(serial["experiments"]["all"]["chosen_scores"]["mean_log_loss"],
                               parallel["experiments"]["all"]["chosen_scores"]["mean_log_loss"], places=9)

    def test_refuses_to_overwrite_output(self) -> None:
        dataset, output = self.root / "ds", self.root / "model"
        write_dataset(dataset, signal=False, markets=10, per_market=4)
        output.mkdir()
        with self.assertRaises(FileExistsError):
            train_baseline(self.small_config(dataset, output))

    def test_price_filter_and_cli(self) -> None:
        dataset, output = self.root / "ds", self.root / "model"
        write_dataset(dataset, signal=False, markets=20, per_market=6)
        code = main([
            "train-baseline", "--dataset", str(dataset), "--output", str(output),
            "--c-grid", "0.1", "--l1-ratios", "0", "--weightings", "none",
            "--filters", "all,price_lt_0.95", "--feature-sets", "price_category", "--jobs", "1",
        ])
        self.assertEqual(code, 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["experiments"]), {"all", "price_lt_0.95"})
        self.assertIn("price_only", manifest["grid"]["feature_sets"])
        self.assertLessEqual(manifest["experiments"]["price_lt_0.95"]["rows"],
                             manifest["experiments"]["all"]["rows"])


if __name__ == "__main__":
    unittest.main()
