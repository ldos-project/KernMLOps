"""Unprivileged offline sensitivity tests; all filesystem fixtures live in /tmp."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import correlate
import numpy as np
import polars as pl
import sensitivity
from polars.testing import assert_frame_equal
from scipy import stats

TARGETS = correlate.TARGETS


def cohort(n_points=16):
    t = np.arange(n_points)
    common = (t - n_points // 2) ** 2 + 10.0
    rows = []
    for replicate in range(30):
        offset = replicate - 7 if replicate < 15 else 0
        for elapsed, base in zip(t, common):
            now = float(base + offset)
            future = float(2 * base + 3 * offset)
            rows.append(
                {
                    "collection_id": f"always-{replicate:02}",
                    "thp": "always",
                    "replicate": replicate,
                    "phase": "run",
                    "elapsed_s": int(elapsed),
                    "f__common": now,
                    "f__other": float(3 * base + offset),
                    "excess_now_mib": now,
                    "excess_1s_mib": future,
                    "excess_change_1s_mib": future - now,
                    "observed_rss_mib": now + 100,
                    "observed_rss_1s_mib": future + 101,
                    "reference_mib": 100.0,
                    "reference_1s_mib": 101.0,
                    "reference_q25_mib": 99.0,
                    "reference_q75_mib": 102.0,
                    "time_ns": int(elapsed) * 1_000_000_000,
                }
            )
    return pl.DataFrame(rows)


def catalog():
    return [
        {
            "name": name,
            "group": "synthetic",
            "scope": "redis",
            "unit": "MiB",
            "description": "Synthetic original feature",
            "availability": "past-only window",
            "limitations": "Synthetic only",
            "target_linked": name == "f__other",
        }
        for name in ("f__common", "f__other")
    ]


def pick(frame, **keys):
    for name, value in keys.items():
        frame = frame.filter(pl.col(name) == value)
    if frame.height != 1:
        raise AssertionError(f"expected one row, got {frame.height}: {keys}")
    return frame.row(0, named=True)


def snapshot(directory):
    return {
        path.relative_to(directory): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_mode,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


class PhaseSensitivityTests(unittest.TestCase):
    def test_nonlinear_common_profile_removed_and_engine_reused(self):
        windows = cohort()
        evaluation = windows.filter(pl.col("replicate") >= 15)
        raw_summary, _ = correlate.summarize_correlations(
            evaluation, catalog(), bootstrap_samples=19
        )
        key = {
            "feature": "f__common",
            "target": TARGETS[0],
            "method": "spearman",
            "adjustment": "adjusted",
        }
        self.assertGreater(pick(raw_summary, **key)["median_r"], 0.99)
        before, entries = windows.clone(), catalog()
        catalog_before = copy.deepcopy(entries)
        with mock.patch.object(
            correlate, "summarize_correlations", wraps=correlate.summarize_correlations
        ) as engine:
            profiles, residuals, summary, per_run, qc = (
                sensitivity.summarize_sensitivity(windows, entries)
            )
        engine.assert_called_once()
        self.assertEqual(
            engine.call_args.kwargs,
            {"seed": 17, "min_points": 8, "min_runs": 10, "bootstrap_samples": 2000},
        )
        self.assertEqual(qc["calibration_runs"], 15)
        self.assertEqual(qc["evaluation_runs"], 15)
        self.assertEqual(set(profiles["feature"]), {"f__common", "f__other", *TARGETS})
        self.assertEqual(profiles["n"].unique().to_list(), [15])
        for name in ("f__common", "f__other", *TARGETS):
            self.assertEqual(residuals[name].unique().to_list(), [0.0])
        row = pick(summary, **key)
        self.assertEqual(row["n_runs"], 0)
        self.assertIsNone(row["median_r"])
        self.assertIsNone(row["p_value"])
        self.assertEqual(per_run["status"].unique().to_list(), ["constant_x"])
        self.assertEqual(set(per_run["replicate"]), set(range(15, 30)))
        untouched = [
            name
            for name in evaluation.columns
            if name not in {"f__common", "f__other", *TARGETS}
        ]
        assert_frame_equal(residuals.select(untouched), evaluation.select(untouched))
        self.assertEqual(residuals["elapsed_s"].dtype, windows["elapsed_s"].dtype)
        assert_frame_equal(windows, before)
        self.assertEqual(entries, catalog_before)

    def test_evaluation_never_and_future_values_cannot_contaminate_profiles(self):
        windows = cohort()
        profiles, residuals, _ = sensitivity.phase_residuals(windows)
        changed_eval = windows.with_columns(
            *[
                pl.when(pl.col("replicate") >= 15)
                .then(pl.col(name) + 1e6)
                .otherwise(pl.col(name))
                .alias(name)
                for name in ("f__common", "f__other", *TARGETS)
            ]
        )
        never = windows.with_columns(
            pl.lit("never").alias("thp"),
            *[
                pl.lit(-1e12).alias(name)
                for name in ("f__common", "f__other", *TARGETS)
            ],
        )
        actual, changed, _ = sensitivity.phase_residuals(
            pl.concat([never, changed_eval, never])
        )
        assert_frame_equal(actual, profiles)
        self.assertEqual(changed["f__common"].unique().to_list(), [1e6])
        self.assertEqual(
            changed["collection_id"].to_list(), residuals["collection_id"].to_list()
        )
        future_only = windows.with_columns(
            pl.when(pl.col("replicate") >= 15)
            .then(pl.col(TARGETS[1]) + 999)
            .otherwise(pl.col(TARGETS[1]))
            .alias(TARGETS[1])
        )
        actual, changed, _ = sensitivity.phase_residuals(future_only)
        assert_frame_equal(actual, profiles)
        for name in ("f__common", "f__other", TARGETS[0], TARGETS[2], "elapsed_s"):
            self.assertTrue(residuals[name].equals(changed[name]))
        # Completed calibration future labels are allowed, but never used to center X or current Y.
        calibration_future = windows.with_columns(
            pl.when(pl.col("replicate") < 15)
            .then(pl.col(TARGETS[1]) + 23)
            .otherwise(pl.col(TARGETS[1]))
            .alias(TARGETS[1])
        )
        actual, changed, _ = sensitivity.phase_residuals(calibration_future)
        self.assertEqual(changed[TARGETS[1]].unique().to_list(), [-23.0])
        assert_frame_equal(
            actual.filter(pl.col("feature") != TARGETS[1]),
            profiles.filter(pl.col("feature") != TARGETS[1]),
        )
        mixed = sensitivity.summarize_sensitivity(
            pl.concat([never, windows, never]), catalog(), bootstrap_samples=13
        )
        expected = sensitivity.summarize_sensitivity(
            windows, catalog(), bootstrap_samples=13
        )
        for left, right in zip(mixed[:4], expected[:4]):
            assert_frame_equal(left, right)

    def test_minimum_eight_distinct_finite_runs_per_variable_no_fill(self):
        windows = (
            cohort()
            .with_columns(
                pl.when(
                    (pl.col("replicate") < 15)
                    & (pl.col("elapsed_s") == 2)
                    & (pl.col("replicate") >= 7)
                )
                .then(None)
                .otherwise(pl.col("f__common"))
                .alias("f__common"),
                pl.when(
                    (pl.col("replicate") < 15)
                    & (pl.col("elapsed_s") == 2)
                    & (pl.col("replicate") >= 8)
                )
                .then(float("inf"))
                .otherwise(pl.col("f__other"))
                .alias("f__other"),
                pl.when((pl.col("replicate") < 15) & (pl.col("elapsed_s") == 3))
                .then(float("nan"))
                .otherwise(pl.col(TARGETS[1]))
                .alias(TARGETS[1]),
                pl.lit(None, dtype=pl.Float64).alias("f__missing"),
            )
            .filter(~((pl.col("replicate") < 15) & (pl.col("elapsed_s") == 4)))
        )
        extra = windows.filter(
            (pl.col("replicate") >= 15) & (pl.col("elapsed_s") == 0)
        ).with_columns(pl.lit(99, dtype=pl.Int64).alias("elapsed_s"))
        windows = pl.concat([windows, extra])
        profiles, residuals, qc = sensitivity.phase_residuals(windows)
        for feature, n in (("f__common", 7), ("f__other", 8)):
            point = pick(profiles, feature=feature, phase="run", elapsed_s=2)
            self.assertEqual(point["n"], n)
            self.assertEqual(point["value"] is None, n < 8)
        missing = profiles.filter(pl.col("feature") == "f__missing")
        self.assertEqual(missing["n"].unique().to_list(), [0])
        self.assertEqual(missing["value"].null_count(), missing.height)
        self.assertEqual(
            pick(profiles, feature=TARGETS[1], phase="run", elapsed_s=3)["n"], 0
        )
        self.assertTrue(
            residuals.filter(pl.col("elapsed_s") == 2)["f__common"].is_null().all()
        )
        self.assertTrue(
            residuals.filter(pl.col("elapsed_s") == 2)["f__other"].is_not_null().all()
        )
        for elapsed in (4, 99):
            self.assertEqual(profiles.filter(pl.col("elapsed_s") == elapsed).height, 0)
            self.assertTrue(
                residuals.filter(pl.col("elapsed_s") == elapsed)["f__common"]
                .is_null()
                .all()
            )
        support = next(
            row for row in qc["profile_support"] if row["feature"] == "f__common"
        )
        self.assertEqual(support["evaluation_finite_input_without_profile"], 3 * 15)
        self.assertEqual(support["evaluation_finite_residuals"], 14 * 15)
        # Duplicating a calibration run cannot turn seven-run support into eight.
        duplicate = windows.filter(
            (pl.col("replicate") == 0) & (pl.col("elapsed_s") == 2)
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sensitivity.phase_residuals(pl.concat([windows, duplicate]))

    def test_phases_and_target_medians_are_independent(self):
        windows = cohort()
        load = windows.with_columns(
            pl.lit("load").alias("phase"),
            *[
                (pl.col(name) + 1000).alias(name)
                for name in ("f__common", "f__other", *TARGETS)
            ],
            pl.lit(500.0).alias("reference_mib"),
        )
        unseen_phase = windows.filter(pl.col("replicate") >= 15).with_columns(
            pl.lit("unseen").alias("phase")
        )
        profiles, residuals, qc = sensitivity.phase_residuals(
            pl.concat([load, windows, unseen_phase])
        )
        for phase, offset in (("run", 0), ("load", 1000)):
            point = pick(profiles, feature=TARGETS[0], phase=phase, elapsed_s=0)
            self.assertEqual(point["value"], 74 + offset)
            self.assertTrue(
                (residuals.filter(pl.col("phase") == phase)[TARGETS[0]] == 0).all()
            )
        self.assertEqual(profiles.filter(pl.col("phase") == "unseen").height, 0)
        self.assertTrue(
            residuals.filter(pl.col("phase") == "unseen")[TARGETS[0]].is_null().all()
        )
        self.assertEqual((qc["calibration_runs"], qc["evaluation_runs"]), (15, 15))
        # Median(future - now) is deliberately not inferred from two other medians.
        changed = windows.with_columns(
            pl.when(pl.col("replicate") < 5)
            .then(0.0)
            .when(pl.col("replicate") < 10)
            .then(10.0)
            .otherwise(11.0)
            .alias(TARGETS[0]),
            pl.when(pl.col("replicate") < 5)
            .then(9.0)
            .when(pl.col("replicate") < 10)
            .then(1.0)
            .otherwise(12.0)
            .alias(TARGETS[1]),
        ).with_columns((pl.col(TARGETS[1]) - pl.col(TARGETS[0])).alias(TARGETS[2]))
        profiles, _, _ = sensitivity.phase_residuals(changed)
        medians = [
            pick(profiles, feature=name, phase="run", elapsed_s=0)["value"]
            for name in TARGETS
        ]
        self.assertEqual(medians, [10.0, 9.0, 1.0])

    def test_adjusted_future_uses_centered_current_and_unchanged_time(self):
        windows = cohort(24)
        rng = np.random.default_rng(31)
        rows = windows.to_dicts()
        for row in rows:
            if row["replicate"] >= 15:
                x, now, future = rng.normal(size=3)
                row["f__common"] += x
                row[TARGETS[0]] += now
                row[TARGETS[1]] += future + 0.8 * now + 0.3 * x
                row[TARGETS[2]] = row[TARGETS[1]] - row[TARGETS[0]]
        windows = pl.DataFrame(rows)
        _, residuals, summary, per_run, _ = sensitivity.summarize_sensitivity(
            windows, catalog(), bootstrap_samples=53
        )
        run = residuals.filter(pl.col("replicate") == 15)
        for target in TARGETS:
            for method in ("pearson", "spearman"):
                columns = ["f__common", target, "elapsed_s"]
                if target != TARGETS[0]:
                    columns.append(TARGETS[0])
                values = run.select(columns).to_numpy()
                if method == "spearman":
                    values = stats.rankdata(values, axis=0, method="average")
                design = np.column_stack([np.ones(run.height), values[:, 2:]])
                residual_xy = (
                    values[:, :2]
                    - design @ np.linalg.lstsq(design, values[:, :2], rcond=None)[0]
                )
                expected = np.corrcoef(residual_xy.T)[0, 1]
                key = {
                    "feature": "f__common",
                    "target": target,
                    "method": method,
                    "adjustment": "adjusted",
                }
                self.assertAlmostEqual(
                    pick(per_run, collection_id="always-15", **key)["r"],
                    expected,
                    places=12,
                )
                aggregate = pick(summary, **key)
                self.assertEqual(aggregate["n_runs"], 15)
                self.assertEqual(
                    aggregate["p_value"],
                    stats.binomtest(
                        aggregate["n_positive"],
                        aggregate["n_positive"] + aggregate["n_negative"],
                        p=0.5,
                    ).pvalue,
                )
                self.assertIn("conditional", aggregate["p_value_method"])
                self.assertIn(
                    "calibration and reference uncertainty", aggregate["ci_method"]
                )
        for frame in (summary, per_run):
            for name in sensitivity.STATISTICS:
                self.assertEqual(
                    frame[name].unique().to_list(), [sensitivity.STATISTICS[name]]
                )
        self.assertIn("raw means correlation", summary["adjustment_notes"][0])
        self.assertIn("NOT a confirmatory holdout", summary["analysis_notes"][0])
        self.assertIn("residual", summary["target_definition"][0])
        self.assertIn("BOTH", summary["reference_notes"][0])
        self.assertNotEqual(
            summary["reference_notes"][0], correlate.NOTES["reference_notes"]
        )
        repeat = sensitivity.summarize_sensitivity(
            windows.reverse(), catalog()[::-1], bootstrap_samples=53
        )
        assert_frame_equal(summary, repeat[2])
        assert_frame_equal(per_run, repeat[3])

    def test_invalid_cohort_and_grid_rejected(self):
        windows = cohort()
        cases = [
            windows.filter(pl.col("replicate") != 14),
            windows.with_columns(
                pl.when(pl.col("replicate") == 29)
                .then(30)
                .otherwise(pl.col("replicate"))
                .alias("replicate")
            ),
            windows.with_columns(pl.lit("same").alias("collection_id")),
            windows.with_columns(pl.col("replicate").cast(pl.Float64)),
            windows.with_columns((pl.col("elapsed_s") + 0.5).alias("elapsed_s")),
            windows.with_columns(pl.lit(None).cast(pl.Int64).alias("elapsed_s")),
            windows.with_columns(pl.lit(None).cast(pl.String).alias("phase")),
            windows.drop(TARGETS[2]),
            windows.drop("f__common", "f__other"),
            pl.concat(
                [
                    windows,
                    windows.filter(pl.col("replicate") == 0).with_columns(
                        pl.lit("second-collection").alias("collection_id")
                    ),
                ]
            ),
        ]
        for frame in cases:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                sensitivity.phase_residuals(frame)
        residuals = sensitivity.phase_residuals(windows)[1]
        with self.assertRaisesRegex(ValueError, "original"):
            sensitivity.phase_residuals(residuals)
        with self.assertRaisesRegex(ValueError, "original"):
            sensitivity.raw_diagnostics(residuals)
        for kwargs in (
            {"seed": -1},
            {"bootstrap_samples": 0},
            {"bootstrap_samples": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sensitivity.summarize_sensitivity(windows, catalog(), **kwargs)

    def test_residual_associations_can_remain_significant_and_missing_horizons_stay_null(
        self,
    ):
        windows = (
            cohort()
            .with_columns(
                *[
                    pl.when(pl.col("replicate") >= 15)
                    .then(pl.col(name) + (pl.col("elapsed_s") % 3))
                    .otherwise(pl.col(name))
                    .alias(name)
                    for name in ("f__common", TARGETS[0])
                ]
            )
            .with_columns(
                pl.when((pl.col("replicate") >= 15) & (pl.col("elapsed_s") == 15))
                .then(None)
                .otherwise(pl.col(TARGETS[1]))
                .alias(TARGETS[1])
            )
        )
        _, residuals, summary, _, qc = sensitivity.summarize_sensitivity(
            windows, catalog(), bootstrap_samples=17
        )
        key = {
            "feature": "f__common",
            "target": TARGETS[0],
            "method": "spearman",
            "adjustment": "adjusted",
        }
        row = pick(summary, **key)
        self.assertAlmostEqual(row["median_r"], 1)
        self.assertEqual(row["n_runs"], 15)
        self.assertEqual(row["p_value"], 2 * 0.5**15)
        self.assertGreater(qc["holm_below_005"], 0)
        self.assertEqual(
            qc["holm_below_005"], summary.filter(pl.col("p_value_holm") < 0.05).height
        )
        self.assertTrue(
            residuals.filter(pl.col("elapsed_s") == 15)[TARGETS[1]].is_null().all()
        )
        sparse = windows.with_columns(
            pl.when((pl.col("replicate") >= 24) & (pl.col("elapsed_s") >= 7))
            .then(float("inf"))
            .otherwise(pl.col("f__common"))
            .alias("f__common")
        )
        _, residuals, summary, per_run, _ = sensitivity.summarize_sensitivity(
            sparse, catalog(), bootstrap_samples=17
        )
        row = pick(summary, **key)
        self.assertEqual(row["n_runs"], 9)
        self.assertEqual(row["status"], "insufficient_runs")
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["p_value"])
        self.assertEqual(
            pick(per_run, collection_id="always-24", **key)["status"],
            "insufficient_points",
        )
        self.assertEqual(residuals["f__common"].null_count(), 6 * 9)

    def test_raw_diagnostics_macro_not_pooled_and_reference_only_changes(self):
        rows = []
        # A dense, zero-error calibration run and sparse, nonzero-error evaluation run.
        for replicate, errors in ((0, [0.0] * 100), (15, [0.0, 4.0])):
            for delta in errors:
                rows.append(
                    {
                        "collection_id": f"a{replicate}",
                        "replicate": replicate,
                        "phase": "run",
                        "thp": "always",
                        "observed_rss_mib": 110.0,
                        "observed_rss_1s_mib": 110.0 + delta,
                        "reference_mib": 100.0,
                        "reference_1s_mib": 101.0,
                        "excess_now_mib": 10.0,
                        "excess_1s_mib": 9.0 + delta,
                    }
                )
        missing = {**rows[-1], "collection_id": "missing", "replicate": 16}
        for name in (
            "observed_rss_mib",
            "observed_rss_1s_mib",
            "reference_mib",
            "reference_1s_mib",
            "excess_now_mib",
            "excess_1s_mib",
        ):
            missing[name] = float("nan")
        windows = pl.DataFrame([*rows, missing])
        before = windows.clone()
        report, per_run = sensitivity.raw_diagnostics(windows)
        zero = pick(per_run, collection_id="a0")
        self.assertEqual(zero["fraction_positive_excess"], 1)
        self.assertEqual(zero["fraction_unchanged_rss"], 1)
        self.assertEqual(zero["fraction_reference_only_change"], 1)
        self.assertEqual(zero["persistence_mae_mib"], 0)
        sparse = pick(per_run, collection_id="a15")
        self.assertEqual(sparse["mean_observed_rss_mib"], 110)
        self.assertEqual(sparse["median_excess_now_mib"], 10)
        self.assertEqual(sparse["fraction_reference_only_change"], 0.5)
        self.assertEqual(sparse["persistence_mae_mib"], 2)
        self.assertAlmostEqual(sparse["persistence_rmse_mib"], np.sqrt(8))
        aggregate = next(
            row for row in report["aggregates"] if row["subset"] == "all_always"
        )
        self.assertEqual(aggregate["n_runs"], 3)
        self.assertEqual(aggregate["metric_run_support"]["persistence_mae_mib"], 2)
        self.assertEqual(aggregate["macro_means"]["persistence_mae_mib"], 1)
        self.assertAlmostEqual(
            aggregate["macro_means"]["persistence_rmse_mib"], np.sqrt(8) / 2
        )
        self.assertEqual(
            aggregate["macro_means"]["fraction_reference_only_change"], 0.75
        )
        self.assertNotAlmostEqual(
            aggregate["macro_means"]["persistence_mae_mib"], 4 / 102
        )
        self.assertIsNone(
            pick(per_run, collection_id="missing")["persistence_rmse_mib"]
        )
        self.assertIn("NOT pooled-window", report["aggregation"])
        self.assertIn("reference-curve artifacts", " ".join(report["limitations"]))
        self.assertIn("NOT validated", " ".join(report["limitations"]))
        json.dumps(report, allow_nan=False)
        assert_frame_equal(windows, before)
        never = windows.with_columns(pl.lit("never").alias("thp"))
        actual, actual_per_run = sensitivity.raw_diagnostics(
            pl.concat([windows, never])
        )
        self.assertEqual(actual, report)
        assert_frame_equal(actual_per_run, per_run)
        load = windows.with_columns(pl.lit("load").alias("phase"))
        separate, _ = sensitivity.raw_diagnostics(pl.concat([windows, load]))
        self.assertEqual(len(separate["aggregates"]), 6)
        self.assertTrue(all(row["n_runs"] <= 3 for row in separate["aggregates"]))


class SensitivityOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bloat-sensitivity-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.object(sensitivity, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.input = self.root / sensitivity.DEFAULT_INPUT
        self.output = self.root / sensitivity.DEFAULT_OUTPUT
        self.input.mkdir(parents=True)
        cohort().write_parquet(self.input / "windows.parquet")
        pl.DataFrame(catalog()).write_parquet(self.input / "feature_catalog.parquet")
        pl.DataFrame({"original_correlations": [1]}).write_parquet(
            self.input / "correlations.parquet"
        )
        (self.input / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": "derived-bloat-correlations",
                    "target_kind": "observed_anonymous_plus_file_rss_excess",
                    "always_collections": 30,
                    "never_reference_collections": 30,
                    "min_reference_runs": 15,
                    "forecast_horizon_seconds": 1,
                    "input_files": {"data/curated/must-not-read.parquet": "not-opened"},
                    "limitations": ["upstream capture losses unknown"],
                }
            )
        )
        for path in self.input.iterdir():
            path.chmod(0o444)
        self.input.chmod(0o555)
        self.addCleanup(self.input.chmod, 0o755)

    def test_end_to_end_readonly_sources_root_relative_and_scoped_outputs(self):
        before = snapshot(self.input)
        publish = sensitivity._publish

        def check_stage(stage, output):
            self.assertFalse(output.exists())
            self.assertEqual(stage.parent, output.parent)
            self.assertEqual(len(list(stage.iterdir())), 11)
            publish(stage, output)

        with mock.patch.object(sensitivity, "_publish", side_effect=check_stage):
            report = sensitivity.analyze(bootstrap_samples=19)
        self.assertEqual(snapshot(self.input), before)
        self.assertEqual(
            report, json.loads((self.output / "manifest.json").read_text())
        )
        self.assertEqual(report["analysis_variant"], sensitivity.VARIANT)
        self.assertEqual(
            report["source_limitations"], ["upstream capture losses unknown"]
        )
        self.assertEqual(report["qc"]["calibration_runs"], 15)
        self.assertEqual(report["qc"]["evaluation_runs"], 15)
        self.assertEqual(report["qc"]["holm_below_005"], 0)
        self.assertEqual(
            set(report["analysis_sources"]),
            {"policy/bloat/sensitivity.py", "policy/bloat/correlate.py"},
        )
        for name, digest in report["analysis_sources"].items():
            source = Path(sensitivity.__file__).resolve().parents[2] / name
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(
            report["input_files"],
            {
                str((self.input / name).relative_to(self.root)): hashlib.sha256(
                    values[0]
                ).hexdigest()
                for name, values in before.items()
            },
        )
        expected_files = {
            "diagnostics.json",
            "manifest.json",
            "per_run_diagnostics.csv",
        }
        for name in (
            "phase_profiles",
            "residual_windows",
            "correlations",
            "correlations_per_run",
        ):
            expected_files.update((f"{name}.csv", f"{name}.parquet"))
            parquet = pl.read_parquet(self.output / f"{name}.parquet")
            csv = pl.read_csv(
                self.output / f"{name}.csv", schema_overrides=parquet.schema
            )
            assert_frame_equal(parquet, csv)
        self.assertEqual({path.name for path in self.output.iterdir()}, expected_files)
        profiles = pl.read_parquet(self.output / "phase_profiles.parquet")
        self.assertEqual(
            profiles.columns, ["feature", "phase", "elapsed_s", "n", "value"]
        )
        residuals = pl.read_parquet(self.output / "residual_windows.parquet")
        self.assertEqual(
            residuals["feature_data_kind"].unique().to_list(),
            ["derived_phase_profile_residuals_not_raw"],
        )
        diagnostics = json.loads((self.output / "diagnostics.json").read_text())
        self.assertEqual(
            diagnostics["input_kind"], "original_uncentered_derived_windows"
        )
        self.assertEqual(len(list(self.output.parent.glob(".phase-sensitivity-*"))), 0)

    def test_output_guards_no_existing_overwrite_or_input_nesting(self):
        self.output.mkdir()
        nonempty = self.output.with_name("nonempty")
        nonempty.mkdir()
        (nonempty / "sentinel").write_text("existing")
        file = self.output.with_name("file")
        file.write_text("existing file")
        link = self.output.with_name("link")
        link.symlink_to(self.output, target_is_directory=True)
        dangling = self.output.with_name("dangling")
        dangling.symlink_to(self.output.with_name("missing"), target_is_directory=True)
        raw = self.root / "data/curated"
        raw.mkdir()
        alias = self.output.with_name("raw-alias")
        alias.symlink_to(raw, target_is_directory=True)
        input_alias = self.output.with_name("input-alias")
        input_alias.symlink_to(self.input, target_is_directory=True)
        before = snapshot(self.root)
        with mock.patch.object(
            pl, "read_parquet", side_effect=AssertionError("must reject before reading")
        ):
            for output in (
                self.output,
                nonempty,
                file,
                link,
                dangling,
                self.input,
                self.input / "nested",
                input_alias / "nested",
                self.input.parent,
                raw / "new",
                alias / "new",
                self.root / "data/derived/bloat-other/new",
                self.root / "data/derived/bloat",
                self.root / "outside",
            ):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(ValueError, "new output directory"),
                ):
                    sensitivity.analyze(self.input, output)
        self.assertEqual(snapshot(self.root), before)
        self.assertTrue(dangling.is_symlink())

    def test_atomic_publish_race_and_failed_staging_leave_no_partial_output(self):
        before = snapshot(self.input)
        with (
            mock.patch.object(
                pl.DataFrame, "write_csv", side_effect=RuntimeError("staging failed")
            ),
            self.assertRaisesRegex(RuntimeError, "staging failed"),
        ):
            sensitivity.analyze(bootstrap_samples=3)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".phase-sensitivity-*")), [])
        publish = sensitivity._publish

        def race(stage, output):
            output.mkdir()
            publish(stage, output)

        with (
            mock.patch.object(sensitivity, "_publish", side_effect=race),
            self.assertRaises(FileExistsError),
        ):
            sensitivity.analyze(bootstrap_samples=3)
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(list(self.output.parent.glob(".phase-sensitivity-*")), [])
        self.assertEqual(snapshot(self.input), before)

    def test_changed_inputs_abort_and_cli_defaults(self):
        digest = sensitivity._digest
        calls = 0

        def changed(path):
            nonlocal calls
            calls += 1
            return "changed" if calls > 6 else digest(path)

        with (
            mock.patch.object(sensitivity, "_digest", side_effect=changed),
            self.assertRaisesRegex(ValueError, "inputs changed"),
        ):
            sensitivity.analyze(bootstrap_samples=3)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".phase-sensitivity-*")), [])
        report = {"output_directory": "fixture", "analysis_notes": "post-hoc", "qc": {}}
        with (
            mock.patch.object(sensitivity, "analyze", return_value=report) as analyze,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            sensitivity.main([])
            analyze.assert_called_once_with(
                input_dir=sensitivity.DEFAULT_INPUT,
                output=sensitivity.DEFAULT_OUTPUT,
                seed=17,
                bootstrap_samples=2000,
            )
            analyze.reset_mock()
            sensitivity.main(
                [
                    "--input-dir",
                    "data/derived/bloat/in",
                    "--output",
                    "data/derived/bloat/out",
                    "--seed",
                    "23",
                    "--bootstrap-samples",
                    "71",
                ]
            )
            analyze.assert_called_once_with(
                input_dir=Path("data/derived/bloat/in"),
                output=Path("data/derived/bloat/out"),
                seed=23,
                bootstrap_samples=71,
            )


if __name__ == "__main__":
    unittest.main()
