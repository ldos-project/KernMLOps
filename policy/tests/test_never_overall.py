import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from plot_never_overall import (
    CONFIGURATIONS,
    METRICS,
    bootstrap_effects,
    plot_never_overall,
)


class NeverOverallTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.rows = [
            {
                "thp": "never",
                "phase": "run",
                "section": "OVERALL",
                "outer_iteration": 0,
                "iteration": 0,
                "mode": mode,
                "metric": metric,
                "unit": unit,
                "collection_id": f"{mode}-{i:02d}",
                "value": float(i + (100 if unit == "ms" else 10)),
            }
            for mode, _, _ in CONFIGURATIONS
            for metric, unit, _, _ in METRICS
            for i in range(30)
        ]

    def save(self):
        pl.DataFrame(self.rows).write_csv(self.directory / "raw_metrics.csv")

    def test_bootstrap_bars_preserve_raw_sd_order_filtering_and_clean_layout(self):
        self.rows += [self.rows[0] | {"thp": "always", "value": 999999.0}]
        self.rows += [self.rows[0] | {"phase": "load", "value": 999999.0}]
        self.save()
        original = (self.directory / "raw_metrics.csv").read_bytes()
        savefig = Figure.savefig
        figures, intervals = [], {}

        def check_plot(figure, *args, **kwargs):
            name = Path(args[0]).name
            figures.append(name)
            self.assertEqual(len(figure.axes), 1)
            self.assertIsNone(figure._suptitle)
            self.assertEqual(figure.texts, [])
            ax = figure.axes[0]
            self.assertEqual(len(ax.patches), 5)
            self.assertLessEqual(ax.get_ylim()[0], 0)
            self.assertGreaterEqual(ax.get_ylim()[1], 0)
            self.assertEqual(list(ax.texts), [])
            for location in ("left", "center", "right"):
                self.assertEqual(ax.get_title(loc=location), "")
            self.assertEqual(
                [label.get_text() for label in ax.get_xticklabels()],
                [item[1] for item in CONFIGURATIONS],
            )
            self.assertTrue(all(bar.get_height() == 0 for bar in ax.patches))
            errors = next(
                item for item in ax.containers if isinstance(item, ErrorbarContainer)
            )
            intervals[name] = [
                segment[:, 1] for segment in errors.lines[2][0].get_segments()
            ]
            self.assertLess(ax.get_ylim()[0], 0)  # Do not clip CIs below zero.
            return savefig(figure, *args, **kwargs)

        with patch.object(Figure, "savefig", autospec=True, side_effect=check_plot):
            summary, outputs = plot_never_overall(
                self.directory, bootstrap_samples=2000
            )
        self.assertEqual(summary.height, 10)
        self.assertEqual(
            set(figures),
            {f"{item[3]}.{suffix}" for item in METRICS for suffix in ("png", "svg")},
        )
        self.assertEqual(len(outputs), 5)
        for metric, unit, _, stem in METRICS:
            rows = summary.filter(pl.col("metric") == metric)
            self.assertEqual(
                rows["mode"].to_list(), [item[0] for item in CONFIGURATIONS]
            )
            values = [i + (100 if unit == "ms" else 10) for i in range(30)]
            self.assertAlmostEqual(rows["mean"][0], statistics.mean(values))
            self.assertAlmostEqual(rows["std"][0], statistics.stdev(values))
            self.assertAlmostEqual(
                rows["sem"][0], statistics.stdev(values) / math.sqrt(30)
            )
            self.assertEqual(
                rows["baseline_mean"].to_list(), [statistics.mean(values)] * 5
            )
            self.assertTrue(all(value == 0 for value in rows["effect_percent"]))
            for row, bounds in zip(
                rows.to_dicts(), intervals[stem + ".png"], strict=True
            ):
                np.testing.assert_allclose(
                    bounds, [row["ci95_low_percent"], row["ci95_high_percent"]]
                )
                if row["mode"] == "none":
                    self.assertEqual(row["ci95_low_percent"], 0)
                    self.assertEqual(row["ci95_high_percent"], 0)
                    self.assertEqual(row["ci95_method"], "reference_identity")
                else:
                    self.assertLess(row["ci95_low_percent"], 0)
                    self.assertGreater(row["ci95_high_percent"], 0)
                    self.assertEqual(
                        row["ci95_method"],
                        "independent_run_bootstrap_percentile_unadjusted",
                    )
        self.assertTrue(all(path.stat().st_size > 0 for path in outputs))
        self.assertEqual((self.directory / "raw_metrics.csv").read_bytes(), original)

    def test_independent_run_resampling_preserves_joint_metrics_and_shared_baseline(
        self,
    ):
        values = np.column_stack((np.arange(30) + 100, np.arange(30) + 10))
        samples = {mode: values.copy() for mode, _, _ in CONFIGURATIONS}
        actual = bootstrap_effects(samples, seed=9, bootstrap_samples=500)
        rng = np.random.default_rng(9)
        baseline = values[rng.integers(0, 30, size=(500, 30))].mean(axis=1)
        for mode, _, _ in CONFIGURATIONS:
            if mode == "none":
                np.testing.assert_array_equal(actual[mode], np.zeros((3, 2)))
                continue
            means = values[rng.integers(0, 30, size=(500, 30))].mean(axis=1)
            draws = (means / baseline - 1) * np.array([100, -100])
            expected = np.quantile(draws, [0.025, 0.975], axis=0)
            np.testing.assert_allclose(actual[mode][1:], expected)
            self.assertTrue(np.all(expected[0] < expected[1]))

    def test_baseline_uncertainty_is_included_and_point_effects_have_correct_sign(self):
        baseline = np.column_stack((np.arange(30) + 10, np.arange(30) + 20))
        samples = {
            mode: np.tile(baseline.mean(axis=0) * [1.2, 0.8], (30, 1))
            for mode, _, _ in CONFIGURATIONS
        }
        samples["none"] = baseline
        for mode, (point, low, high) in bootstrap_effects(samples).items():
            if mode == "none":
                np.testing.assert_array_equal(point, [0, 0])
            else:
                np.testing.assert_allclose(point, [20, 20])
                self.assertTrue(np.all(high > low))  # Numerator is constant.

    def test_constant_samples_and_speedups_are_valid(self):
        samples = {mode: np.full((30, 2), 42.0) for mode, _, _ in CONFIGURATIONS}
        for result in bootstrap_effects(samples).values():
            np.testing.assert_array_equal(result, np.zeros((3, 2)))
        samples["all"][:] = [21, 84]
        point, low, high = bootstrap_effects(samples)["all"]
        np.testing.assert_allclose(point, [-50, -100])
        np.testing.assert_allclose(low, point)
        np.testing.assert_allclose(high, point)

    def test_seed_and_input_row_order_are_reproducible(self):
        self.save()
        first, _ = plot_never_overall(self.directory, seed=11, bootstrap_samples=500)
        self.rows.reverse()
        self.save()
        second, _ = plot_never_overall(self.directory, seed=11, bootstrap_samples=500)
        self.assertEqual(first.to_dicts(), second.to_dicts())

    def test_missing_duplicate_nonpositive_nonfinite_and_wrong_units_are_rejected(self):
        original = self.rows.copy()
        for mutation in (
            "missing",
            "duplicate",
            "nan",
            "zero",
            "negative",
            "unit",
            "repeat",
        ):
            self.rows = [row.copy() for row in original]
            if mutation == "missing":
                self.rows.pop(0)
            elif mutation == "duplicate":
                self.rows[0]["collection_id"] = self.rows[1]["collection_id"]
            elif mutation in ("nan", "zero", "negative"):
                self.rows[0]["value"] = {
                    "nan": float("nan"),
                    "zero": 0,
                    "negative": -1,
                }[mutation]
            elif mutation == "unit":
                self.rows[0]["unit"] = "seconds"
            else:
                self.rows[0]["iteration"] = 1
            self.save()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plot_never_overall(self.directory)
            self.assertFalse((self.directory / "never_runtime_slowdown.png").exists())

    def test_output_links_cannot_overwrite_input(self):
        self.save()
        source = self.directory / "raw_metrics.csv"
        (self.directory / "never_slowdown_throughput_loss.csv").symlink_to(source)
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            plot_never_overall(self.directory)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
