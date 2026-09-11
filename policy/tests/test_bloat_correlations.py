import unittest

import numpy as np
import polars as pl
from correlate import PER_RUN_SCHEMA, SUMMARY_SCHEMA, TARGETS, summarize_correlations
from polars.testing import assert_frame_equal
from scipy import stats


def catalog(*names, linked=()):
    return [
        dict(
            name=name,
            group="rss" if name in linked else "events",
            scope="redis",
            unit="MiB" if name in linked else "count",
            description=f"Synthetic {name}",
            target_linked=name in linked,
            availability="past-only window",
            limitations="Synthetic telemetry only",
        )
        for name in names
    ]


def run_frame(cid, x, now, *, future=None, elapsed=None, phase="run", thp="always"):
    x, now = np.asarray(x, dtype=float), np.asarray(now, dtype=float)
    future = now.copy() if future is None else np.asarray(future, dtype=float)
    n = len(x)
    return pl.DataFrame(
        {
            "collection_id": [cid] * n,
            "thp": [thp] * n,
            "replicate": [0] * n,
            "phase": [phase] * n,
            "elapsed_s": np.linspace(0, 1, n) if elapsed is None else elapsed,
            "time_ns": np.arange(n),
            "redis_pid": [123] * n,
            "f__x": x,
            "excess_now_mib": now,
            "excess_1s_mib": future,
            "excess_change_1s_mib": future - now,
        }
    )


def cohort(n_runs=20, n_points=40):
    rng = np.random.default_rng(102)
    frames = []
    for i in range(n_runs):
        x = rng.normal(size=n_points)
        t = np.linspace(0, 1, n_points)
        now = x + 0.4 * t + 0.5 * rng.normal(size=n_points)
        future = now + 0.7 * x + 0.3 * t + 0.3 * rng.normal(size=n_points)
        frames.append(run_frame(f"run-{i:02}", x, now, future=future))
    return (
        pl.concat(frames)
        .rename({"f__x": "f__positive"})
        .with_columns((-pl.col("f__positive")).alias("f__negative"))
    )


def pick(
    frame,
    *,
    feature="f__x",
    target=TARGETS[0],
    method="pearson",
    adjustment="raw",
    phase="run",
):
    return frame.filter(
        (pl.col("feature") == feature)
        & (pl.col("target") == target)
        & (pl.col("method") == method)
        & (pl.col("adjustment") == adjustment)
        & (pl.col("phase") == phase)
    ).row(0, named=True)


def manual_partial(x, y, controls, method):
    values = np.column_stack((x, y, *controls))
    values = values[np.isfinite(values).all(axis=1)]
    if method == "spearman":
        values = np.column_stack(
            [stats.rankdata(col, method="average") for col in values.T]
        )
    design = np.column_stack((np.ones(len(values)), values[:, 2:]))
    residuals = (
        values[:, :2] - design @ np.linalg.lstsq(design, values[:, :2], rcond=None)[0]
    )
    return np.corrcoef(residuals.T)[0, 1]


class BloatCorrelationsTest(unittest.TestCase):
    def test_twenty_runs_positive_negative_and_catalog_metadata(self):
        windows = cohort()
        entries = catalog("f__positive", "f__negative", linked=("f__positive",))
        original = windows.clone()
        catalog_before = [entry.copy() for entry in entries]
        summary, per_run = summarize_correlations(windows, entries)
        self.assertEqual(summary.height, 2 * 3 * 2 * 2)
        self.assertEqual(per_run.height, summary.height * 20)
        for feature, sign in (("f__positive", 1), ("f__negative", -1)):
            for target in TARGETS:
                for method in ("pearson", "spearman"):
                    row = pick(summary, feature=feature, target=target, method=method)
                    self.assertGreater(sign * row["median_r"], 0.7)
                    self.assertEqual(row["n_runs"], 20)
                    self.assertEqual(
                        row["n_positive" if sign > 0 else "n_negative"], 20
                    )
                    self.assertEqual(row["n_tied"], 0)
                    self.assertEqual(row["n_total_windows"], 800)
                    self.assertEqual(row["median_windows_per_run"], 40)
                    self.assertAlmostEqual(row["p_value"], 2 * 0.5**20)
                    self.assertEqual(
                        row["p_value_method"], "two-sided independent-run sign test"
                    )
                    self.assertIn("Directional consistency", row["p_value_null"])
                    self.assertIn("Not pooled Pearson", row["p_value_null"])
                    self.assertIn("Welch", row["p_value_null"])
        linked = summary.filter(pl.col("feature") == "f__positive").to_dicts()
        for row in linked:
            for key, value in entries[0].items():
                if key != "name":
                    self.assertEqual(row[key], value)
            self.assertIn("not an independent predictor", row["predictor_role"])
            self.assertIn("exclude reference uncertainty", row["reference_notes"])
            self.assertIn("minimum support 15", row["reference_notes"])
            self.assertIn("IQR separately", row["reference_notes"])
            self.assertIn("no test-trained claim", row["reference_notes"])
            self.assertIn("not full RSS", row["target_definition"])
            self.assertIn("whole collection phase", row["adjustment_notes"])
            self.assertIn("not deployable", row["adjustment_notes"])
            self.assertIn("predictive validation", row["adjustment_notes"])
            self.assertIn("whole independent-run", row["ci_method"])
        assert_frame_equal(windows, original)
        self.assertEqual(entries, catalog_before)

    def test_run_sign_test_and_equal_weight_not_dense_window_inference(self):
        x = np.tile([-1.0, 0, 1, 0], 2)
        runs = [run_frame(f"run-{i:02}", x, x if i < 15 else -x) for i in range(20)]
        sparse, _ = summarize_correlations(
            pl.concat(runs), catalog("f__x"), bootstrap_samples=199
        )
        dense = pl.concat([*runs[:-1], *([runs[-1]] * 500)])
        summary, per_run = summarize_correlations(
            dense, catalog("f__x"), bootstrap_samples=199
        )
        row, before = pick(summary), pick(sparse)
        self.assertEqual(row["n_runs"], 20)
        self.assertEqual((row["n_positive"], row["n_negative"]), (15, 5))
        self.assertAlmostEqual(row["median_r"], 1)
        self.assertAlmostEqual(row["mean_r"], 0.5)
        self.assertEqual(row["median_windows_per_run"], 8)
        self.assertEqual(row["n_total_windows"], 8 * (19 + 500))
        self.assertEqual(row["p_value"], stats.binomtest(15, 20, p=0.5).pvalue)
        for field in (
            "median_r",
            "mean_r",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_holm",
        ):
            self.assertAlmostEqual(row[field], before[field])
        pooled = stats.pearsonr(dense["f__x"].to_numpy(), dense[TARGETS[0]].to_numpy())
        self.assertLess(pooled.statistic, -0.9)
        self.assertLess(pooled.pvalue, row["p_value"] / 100)
        self.assertEqual(per_run["collection_id"].n_unique(), 20)

    def test_whole_phase_time_and_current_excess_adjustment(self):
        t = np.linspace(0, 1, 64)
        q, _ = np.linalg.qr(
            np.column_stack(
                (np.ones(64), t, np.sin(6 * t), np.cos(6 * t), np.sin(12 * t))
            )
        )
        now = 6 * q[:, 2] + 2 * q[:, 1]
        x = 7 * q[:, 1] + 8 * now + q[:, 3]
        future = 11 * q[:, 1] + 9 * now + q[:, 4]
        summary, per_run = summarize_correlations(
            run_frame("a", x, now, future=future, elapsed=t), catalog("f__x")
        )
        self.assertGreater(pick(per_run)["r"], 0.95)
        self.assertGreater(pick(per_run, adjustment="adjusted")["r"], 0.99)
        self.assertEqual(pick(summary, adjustment="adjusted")["controls"], "elapsed_s")
        for target, y in ((TARGETS[1], future), (TARGETS[2], future - now)):
            row = pick(per_run, target=target, adjustment="adjusted")
            self.assertGreater(manual_partial(x, y, [t], "pearson"), 0.99)
            self.assertAlmostEqual(
                row["r"], manual_partial(x, y, [t, now], "pearson"), places=12
            )
            self.assertLess(abs(row["r"]), 1e-10)
            self.assertEqual(
                pick(summary, target=target, adjustment="adjusted")["controls"],
                "elapsed_s,excess_now_mib",
            )
        x, now = 30 * q[:, 1] + q[:, 2], 40 * q[:, 1] + q[:, 3]
        _, per_run = summarize_correlations(
            run_frame("b", x, now, elapsed=t), catalog("f__x")
        )
        self.assertGreater(pick(per_run)["r"], 0.99)
        self.assertLess(abs(pick(per_run, adjustment="adjusted")["r"]), 1e-10)

    def test_spearman_average_ties_and_ranked_controls_before_ols(self):
        rng = np.random.default_rng(82)
        t = np.repeat(np.arange(10, dtype=float), 5)
        now = rng.integers(0, 10, 50).astype(float)
        x = t + now + rng.integers(0, 15, 50)
        future = 2 * t + 0.5 * now + x + rng.integers(0, 20, 50)
        frame = run_frame("a", x, now, future=future, elapsed=t)
        _, per_run = summarize_correlations(frame, catalog("f__x"))
        for target in TARGETS:
            y = frame[target].to_numpy()
            raw = pick(per_run, target=target, method="spearman")
            self.assertAlmostEqual(raw["r"], stats.spearmanr(x, y).statistic, places=12)
            for method in ("pearson", "spearman"):
                controls = [t] if target == TARGETS[0] else [t, now]
                adjusted = pick(
                    per_run, target=target, method=method, adjustment="adjusted"
                )
                self.assertEqual(adjusted["status"], "ok")
                self.assertAlmostEqual(
                    adjusted["r"], manual_partial(x, y, controls, method), places=12
                )

    def test_never_excluded_even_with_labels_and_matching_collection_ids(self):
        always = cohort()
        never = always.with_columns(
            pl.lit("never").alias("thp"),
            *[(-1000 * pl.col(target)).alias(target) for target in TARGETS],
        )
        expected = summarize_correlations(
            always, catalog("f__positive"), bootstrap_samples=71
        )
        mixed = pl.concat([never, always, never])
        before = mixed.clone()
        actual = summarize_correlations(
            mixed, catalog("f__positive"), bootstrap_samples=71
        )
        for left, right in zip(expected, actual):
            assert_frame_equal(left, right)
        assert_frame_equal(mixed, before)
        empty = summarize_correlations(never.drop(TARGETS), catalog("f__positive"))
        self.assertEqual([frame.height for frame in empty], [0, 0])
        self.assertEqual(empty[0].schema, SUMMARY_SCHEMA)
        self.assertEqual(empty[1].schema, PER_RUN_SCHEMA)

    def test_finite_complete_cases_and_differing_adjusted_coverage(self):
        frame = (
            cohort(n_runs=1, n_points=16)
            .rename({"f__positive": "f__x"})
            .drop("f__negative")
        )
        rows = frame.to_dicts()
        for index, column, value in (
            (0, "f__x", float("nan")),
            (1, "f__x", float("inf")),
            (2, TARGETS[0], None),
            (3, TARGETS[0], -float("inf")),
            (4, "elapsed_s", float("nan")),
            (5, TARGETS[1], float("nan")),
            (6, TARGETS[2], None),
        ):
            rows[index][column] = value
        frame = pl.DataFrame(rows)
        summary, per_run = summarize_correlations(frame, catalog("f__x"), min_points=11)
        for method in ("pearson", "spearman"):
            for target, raw_n, adjusted_n in (
                (TARGETS[0], 12, 11),
                (TARGETS[1], 13, 10),
                (TARGETS[2], 13, 10),
            ):
                raw = pick(per_run, target=target, method=method)
                adjusted = pick(
                    per_run, target=target, method=method, adjustment="adjusted"
                )
                self.assertEqual(raw["n_points"], raw_n)
                self.assertEqual(adjusted["n_points"], adjusted_n)
                self.assertEqual(adjusted["n_windows"], 16)
                if adjusted_n < 11:
                    self.assertIsNone(adjusted["r"])
                    self.assertEqual(adjusted["status"], "insufficient_points")
                else:
                    self.assertEqual(adjusted["status"], "ok")
                    self.assertAlmostEqual(
                        adjusted["r"],
                        manual_partial(
                            frame["f__x"].to_numpy(),
                            frame[target].to_numpy(),
                            [frame["elapsed_s"].to_numpy()],
                            method,
                        ),
                        places=12,
                    )
                aggregate = pick(
                    summary, target=target, method=method, adjustment="adjusted"
                )
                self.assertEqual(aggregate["n_complete_windows"], adjusted_n)
                self.assertEqual(aggregate["n_input_windows"], 16)
                self.assertEqual(
                    aggregate["n_total_windows"], adjusted_n if adjusted_n >= 11 else 0
                )

    def test_constants_missing_columns_and_minimum_points_keep_qc(self):
        base = cohort(n_runs=1).rename({"f__positive": "f__x"}).drop("f__negative")
        frames = [
            base,
            base.with_columns(
                pl.lit("constant-x").alias("collection_id"), pl.lit(0.0).alias("f__x")
            ),
            base.with_columns(
                pl.lit("constant-y").alias("collection_id"),
                pl.lit(1.0).alias(TARGETS[0]),
            ),
            base.with_columns(
                pl.lit("missing-x").alias("collection_id"),
                pl.lit(None, dtype=pl.Float64).alias("f__x"),
            ),
            base[:7].with_columns(pl.lit("short").alias("collection_id")),
        ]
        summary, per_run = summarize_correlations(
            pl.concat(frames), catalog("f__x", "f__absent")
        )
        raw = per_run.filter(
            (pl.col("feature") == "f__x")
            & (pl.col("target") == TARGETS[0])
            & (pl.col("method") == "pearson")
            & (pl.col("adjustment") == "raw")
        )
        statuses = {row["collection_id"]: row["status"] for row in raw.to_dicts()}
        self.assertEqual(
            statuses,
            {
                "run-00": "ok",
                "constant-x": "constant_x",
                "constant-y": "constant_y",
                "missing-x": "insufficient_points",
                "short": "insufficient_points",
            },
        )
        row = pick(summary)
        self.assertEqual(
            (row["n_runs"], row["n_candidate_runs"], row["n_excluded_runs"]), (1, 5, 4)
        )
        self.assertEqual(row["n_tied"], 0)
        self.assertEqual(row["n_total_windows"], 40)
        self.assertEqual(row["n_complete_windows"], 127)
        absent = pick(per_run, feature="f__absent")
        self.assertEqual(
            (absent["n_points"], absent["r"], absent["status"]),
            (0, None, "missing_columns"),
        )
        self.assertIn("f__absent", absent["reason"])
        summary, per_run = summarize_correlations(
            base.drop("elapsed_s", TARGETS[1]), []
        )
        self.assertEqual(pick(per_run)["status"], "ok")
        self.assertEqual(
            pick(per_run, adjustment="adjusted")["status"], "missing_columns"
        )
        self.assertEqual(pick(per_run, target=TARGETS[1])["status"], "missing_columns")
        self.assertIsNone(pick(summary)["target_linked"])
        self.assertIn("Unknown", pick(summary)["predictor_role"])

    def test_degenerate_controls_residuals_and_residual_dof(self):
        t = np.arange(16, dtype=float)
        x, y = np.sin(t), np.cos(t)
        cases = [
            (
                run_frame("a", x, y, elapsed=np.ones(16)),
                TARGETS[0],
                "degenerate_controls",
            ),
            (
                run_frame("a", x, 2 * t, future=y, elapsed=t),
                TARGETS[1],
                "degenerate_controls",
            ),
            (run_frame("a", 2 * t, y, elapsed=t), TARGETS[0], "constant_residual_x"),
            (run_frame("a", x, 2 * t, elapsed=t), TARGETS[0], "constant_residual_y"),
            (run_frame("a", x[:3], y[:3]), TARGETS[0], "insufficient_residual_dof"),
            (
                run_frame("a", x[:4], y[:4], future=t[:4]),
                TARGETS[1],
                "insufficient_residual_dof",
            ),
        ]
        for frame, target, status in cases:
            with self.subTest(status=status, target=target):
                _, per_run = summarize_correlations(
                    frame, catalog("f__x"), min_points=2
                )
                for method in ("pearson", "spearman"):
                    row = pick(
                        per_run, target=target, method=method, adjustment="adjusted"
                    )
                    self.assertEqual(row["status"], status)
                    self.assertIsNone(row["r"])
        frame = run_frame("a", x, x, future=y, elapsed=t)
        _, per_run = summarize_correlations(frame, catalog("f__x"))
        self.assertEqual(
            pick(per_run, target=TARGETS[1], adjustment="adjusted")["status"],
            "constant_residual_x",
        )

    def test_insufficient_runs_descriptive_even_with_many_windows(self):
        frame = cohort(n_runs=9, n_points=400)
        summary, _ = summarize_correlations(frame, catalog("f__positive"))
        for row in summary.to_dicts():
            self.assertEqual(row["status"], "insufficient_runs")
            self.assertEqual(row["n_runs"], 9)
            self.assertIsNotNone(row["median_r"])
            for field in ("ci_low", "ci_high", "p_value", "p_value_holm"):
                self.assertIsNone(row[field])
            self.assertEqual(row["holm_family_size"], 0)
        summary, _ = summarize_correlations(cohort(), [], min_runs=21)
        self.assertEqual(summary["status"].unique().to_list(), ["insufficient_runs"])

    def test_phases_separate_collections_not_replicate_pid_or_phase_pseudoreplicates(
        self,
    ):
        run = cohort(n_runs=5)
        load = run.with_columns(
            pl.lit("load").alias("phase"), (-pl.col("f__positive")).alias("f__positive")
        )
        summary, per_run = summarize_correlations(pl.concat([load, run]), [])
        self.assertEqual(summary.height, 2 * 2 * 3 * 2 * 2)
        self.assertEqual(per_run["collection_id"].n_unique(), 5)
        for phase, sign in (("run", 1), ("load", -1)):
            row = pick(summary, feature="f__positive", phase=phase)
            self.assertEqual(row["n_runs"], 5)
            self.assertEqual(row["status"], "insufficient_runs")
            self.assertGreater(sign * row["median_r"], 0.8)
        expected = summarize_correlations(run, [])
        for frame in (run.drop("phase"), run.with_columns(pl.lit(None).alias("phase"))):
            actual = summarize_correlations(frame, [])
            for left, right in zip(expected, actual):
                assert_frame_equal(left, right)

    def test_exact_zero_ties_excluded_but_near_zero_signs_retained(self):
        x = np.tile([-1.0, 0, 1, 0], 2)
        orthogonal = np.tile([0.0, -1, 0, 1], 2)
        frames = [
            run_frame(f"a-{i:02}", x, x if i < 10 else -x if i < 15 else orthogonal)
            for i in range(20)
        ]
        summary, _ = summarize_correlations(pl.concat(frames), [], bootstrap_samples=31)
        row = pick(summary)
        self.assertEqual(
            (row["n_positive"], row["n_negative"], row["n_tied"]), (10, 5, 5)
        )
        self.assertEqual(row["p_value"], stats.binomtest(10, 15, p=0.5).pvalue)
        for perturbation in (0.0, 1e-15):
            frames = [
                run_frame(f"a-{i:02}", x, orthogonal + perturbation * x)
                for i in range(20)
            ]
            summary, _ = summarize_correlations(
                pl.concat(frames), [], bootstrap_samples=31
            )
            row = pick(summary)
            if perturbation == 0:
                self.assertEqual(
                    (row["ci_low"], row["ci_high"], row["n_tied"]), (0.0, 0.0, 20)
                )
                self.assertEqual(row["status"], "no_nonzero_runs")
                self.assertIsNone(row["p_value"])
                self.assertIsNone(row["p_value_holm"])
            else:
                self.assertEqual((row["n_positive"], row["n_tied"]), (20, 0))
                self.assertAlmostEqual(row["p_value"], 2 * 0.5**20)

    def test_bootstrap_seed_order_bounds_and_whole_run_reference(self):
        frame = cohort()
        entries = catalog("f__positive", "f__negative")
        state = np.random.get_state()
        summary, per_run = summarize_correlations(
            frame, entries, seed=17, bootstrap_samples=4001
        )
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        reordered = summarize_correlations(
            frame.select(frame.columns[::-1]),
            entries[::-1],
            seed=17,
            bootstrap_samples=4001,
        )
        assert_frame_equal(summary, reordered[0])
        assert_frame_equal(per_run, reordered[1])
        first = summary.row(0, named=True)
        rs = per_run.filter(
            (pl.col("feature") == first["feature"])
            & (pl.col("target") == first["target"])
            & (pl.col("method") == first["method"])
            & (pl.col("adjustment") == first["adjustment"])
        )["r"].to_numpy()
        draws = np.random.default_rng(17).integers(0, len(rs), size=(4001, len(rs)))
        low, high = np.quantile(np.median(rs[draws], axis=1), [0.025, 0.975])
        self.assertEqual((first["ci_low"], first["ci_high"]), (low, high))
        for row in summary.to_dicts():
            self.assertTrue(-1 <= row["ci_low"] <= row["ci_high"] <= 1)
        changed, _ = summarize_correlations(
            frame, entries, seed=18, bootstrap_samples=4001
        )
        self.assertFalse(
            summary.select("ci_low", "ci_high").equals(
                changed.select("ci_low", "ci_high")
            )
        )

    def test_holm_manual_full_family_includes_all_variants_and_linked_features(self):
        frame = cohort().with_columns(
            (pl.col("f__positive") + 0.05 * pl.col(TARGETS[0])).alias("f__diagnostic"),
            pl.lit(0.0).alias("f__constant"),
        )
        entries = catalog(
            "f__positive",
            "f__negative",
            "f__diagnostic",
            "f__constant",
            linked=("f__diagnostic",),
        )
        summary, _ = summarize_correlations(frame, entries, bootstrap_samples=53)
        valid = summary.filter(pl.col("p_value").is_not_null()).to_dicts()
        self.assertEqual(len(valid), 3 * 3 * 2 * 2)
        self.assertEqual(
            {row["feature"] for row in valid},
            {"f__positive", "f__negative", "f__diagnostic"},
        )
        self.assertEqual({row["target"] for row in valid}, set(TARGETS))
        self.assertEqual({row["method"] for row in valid}, {"pearson", "spearman"})
        self.assertEqual({row["adjustment"] for row in valid}, {"raw", "adjusted"})
        ordered = sorted(valid, key=lambda row: row["p_value"])
        previous = 0
        for rank, row in enumerate(ordered):
            expected = min(1, max(previous, (len(valid) - rank) * row["p_value"]))
            self.assertAlmostEqual(row["p_value_holm"], expected)
            self.assertGreaterEqual(row["p_value_holm"], row["p_value"])
            previous = expected
        self.assertEqual(summary["holm_family_size"].unique().to_list(), [len(valid)])
        self.assertEqual(ordered[0]["p_value_holm"], len(valid) * ordered[0]["p_value"])
        invalid = summary.filter(pl.col("feature") == "f__constant")
        self.assertEqual(invalid["p_value_holm"].null_count(), invalid.height)
        load = frame.with_columns(pl.lit("load").alias("phase"))
        both, _ = summarize_correlations(
            pl.concat([frame, load]), entries, bootstrap_samples=11
        )
        self.assertEqual(both["holm_family_size"].unique().to_list(), [2 * len(valid)])
        self.assertEqual(pick(both, feature="f__positive")["n_runs"], 20)

    def test_empty_and_all_unsupported_have_stable_typed_schemas(self):
        for frame in (
            pl.DataFrame(),
            cohort().clear(),
            cohort().drop("f__positive", "f__negative"),
        ):
            summary, per_run = summarize_correlations(frame, [])
            self.assertEqual((summary.height, per_run.height), (0, 0))
            self.assertEqual(summary.schema, SUMMARY_SCHEMA)
            self.assertEqual(per_run.schema, PER_RUN_SCHEMA)
        for expression in (
            pl.lit(0.0),
            pl.lit(None, dtype=pl.Float64),
            pl.lit(float("nan")),
        ):
            frame = (
                cohort(n_runs=1)
                .drop("f__negative")
                .with_columns(expression.alias("f__positive"))
            )
            summary, per_run = summarize_correlations(frame, [])
            self.assertEqual(summary.height, 12)
            self.assertEqual(summary.schema, SUMMARY_SCHEMA)
            self.assertEqual(per_run.schema, PER_RUN_SCHEMA)
            self.assertEqual(summary["n_runs"].to_list(), [0] * 12)
            self.assertEqual(summary["median_r"].null_count(), 12)
            self.assertEqual(summary["n_tied"].to_list(), [0] * 12)
            self.assertEqual(per_run["r"].null_count(), 12)

    def test_finite_extreme_values_do_not_overflow(self):
        x = np.tile([-1e308, 0, 1e308, 0], 2)
        with np.errstate(all="raise"):
            _, per_run = summarize_correlations(run_frame("a", x, -x), [])
        self.assertEqual(pick(per_run)["status"], "ok")
        self.assertAlmostEqual(pick(per_run)["r"], -1)

    def test_invalid_configuration_and_ambiguous_run_identity_rejected(self):
        for kwargs in (
            {"min_points": 1},
            {"min_runs": 9},
            {"bootstrap_samples": 0},
            {"bootstrap_samples": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize_correlations(pl.DataFrame(), [], **kwargs)
        for frame in (
            cohort(n_runs=1).drop("thp"),
            cohort(n_runs=1).drop("collection_id"),
            cohort(n_runs=1).with_columns(pl.lit(None).alias("collection_id")),
        ):
            with self.assertRaises(ValueError):
                summarize_correlations(frame, [])
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_correlations(
                cohort(n_runs=1), catalog("f__positive", "f__positive")
            )


if __name__ == "__main__":
    unittest.main()
