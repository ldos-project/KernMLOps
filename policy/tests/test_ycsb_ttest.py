import importlib.util
import statistics
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import polars as pl
from scipy import stats

SCRIPT = Path(__file__).resolve().parents[1] / "ycsb_ttest.py"
SPEC = importlib.util.spec_from_file_location("ycsb_ttest_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
welch_ttest = MODULE.welch_ttest
compare_ycsb_runs = MODULE.compare_ycsb_runs

KEYS = (
    "benchmark_name",
    "phase",
    "outer_iteration",
    "iteration",
    "section",
    "metric",
    "unit",
)
COUNTS = ("left_missing", "right_missing", "left_invalid", "right_invalid")
STAT_FIELDS = {
    "left_n",
    "right_n",
    "left_mean",
    "right_mean",
    "left_std",
    "right_std",
    "difference",
    "change_percent",
    "t_statistic",
    "p_value",
    "status",
}
INVOCATION_SCHEMA = {
    "collection_id": pl.String,
    "invocation_id": pl.UInt32,
    "benchmark_name": pl.String,
    "run_label": pl.String,
    "phase": pl.String,
    "iteration": pl.UInt32,
    "outer_iteration": pl.UInt32,
    "argv": pl.List(pl.String),
    "start_monotonic_ns": pl.UInt64,
    "end_monotonic_ns": pl.UInt64,
    "exit_code": pl.Int32,
    "execution_state": pl.String,
    "parse_state": pl.String,
    "error": pl.String,
    "stdout_path": pl.String,
    "stderr_path": pl.String,
}
METRIC_SCHEMA = {
    "collection_id": pl.String,
    "invocation_id": pl.UInt32,
    "source": pl.String,
    "line_number": pl.UInt64,
    "section": pl.String,
    "metric": pl.String,
    "value_raw": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
}


class YCSBTTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def make_run(self, name, value=10.0, *, cid=None, samples=None):
        directory = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.root))
        samples = [{"value": value}] if samples is None else samples
        invocations, metrics = [], []
        for index, sample in enumerate(samples or [{}]):
            phase = sample.get("phase", "run")
            invocation = {
                "collection_id": cid or name,
                "invocation_id": index,
                "run_label": name,
                "benchmark_name": "ycsb",
                "phase": phase,
                "iteration": 0,
                "outer_iteration": 0,
                "argv": [
                    "python3",
                    f"/{name}/bin/ycsb",
                    phase,
                    "basic",
                    "-p",
                    "recordcount=100",
                ],
                "start_monotonic_ns": index * 100,
                "end_monotonic_ns": index * 100 + 1,
                "exit_code": 0,
                "execution_state": "completed",
                "parse_state": "ok",
                "error": None,
                "stdout_path": f"/{name}/stdout.txt",
                "stderr_path": f"/{name}/stderr.txt",
            }
            invocation.update(
                {key: val for key, val in sample.items() if key in INVOCATION_SCHEMA}
            )
            invocations.append(invocation)
            sample_value = sample.get("value", value)
            metrics.append(
                {
                    "collection_id": cid or name,
                    "invocation_id": invocation["invocation_id"],
                    "source": "stdout",
                    "line_number": index + 1,
                    "value_raw": None if sample_value is None else str(sample_value),
                    "section": "OVERALL",
                    "metric": "Throughput",
                    "value": value,
                    "unit": "ops/sec",
                    **{key: val for key, val in sample.items() if key in METRIC_SCHEMA},
                }
            )
        if not samples:
            metrics.clear()  # A captured invocation may emit no metrics.
        pl.DataFrame(invocations, schema=INVOCATION_SCHEMA).write_parquet(
            directory / "ycsb_invocations.end.parquet"
        )
        pl.DataFrame(metrics, schema=METRIC_SCHEMA).write_parquet(
            directory / "ycsb_metrics.end.parquet"
        )
        return directory

    def test_welch_matches_scipy_with_generators(self):
        for left, right in [
            ([1.0, 2.0, 4.0, 8.0], [11.0, 15.0, 30.0, 60.0, 90.0]),
            ([2.0, 2.0, 2.0], [1.0, 4.0, 8.0, 12.0]),
            ([1.0, 4.0, 8.0, 12.0], [2.0, 2.0, 2.0]),
        ]:
            with self.subTest(left=left, right=right):
                result = welch_ttest((x for x in left), (x for x in right))
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Precision loss occurred.*", RuntimeWarning
                    )
                    reference = stats.ttest_ind(
                        right, left, equal_var=False, alternative="two-sided"
                    )
                self.assertTrue(STAT_FIELDS <= result.keys())
                self.assertEqual(result["status"], "ok")
                for side, values in (("left", left), ("right", right)):
                    self.assertEqual(result[f"{side}_n"], len(values))
                    self.assertAlmostEqual(
                        result[f"{side}_mean"], statistics.mean(values)
                    )
                    self.assertAlmostEqual(
                        result[f"{side}_std"], statistics.stdev(values)
                    )
                self.assertAlmostEqual(
                    result["difference"], statistics.mean(right) - statistics.mean(left)
                )
                change = 100 * (statistics.mean(right) / statistics.mean(left) - 1)
                self.assertAlmostEqual(result["change_percent"], change)
                self.assertAlmostEqual(
                    result["t_statistic"], reference.statistic, places=12
                )
                self.assertAlmostEqual(result["p_value"], reference.pvalue, places=12)

    def test_welch_degenerate_samples(self):
        for left, right, status in [
            ([], [], "insufficient_samples"),
            ([], [2.0, 3.0], "insufficient_samples"),
            ([1.0], [2.0], "insufficient_samples"),
            ([1.0], [2.0, 3.0], "insufficient_samples"),
            ([2.0, 2.0], [2.0, 2.0], "zero_variance"),
            ([2.0, 2.0], [5.0, 5.0], "zero_variance"),
        ]:
            with self.subTest(left=left, right=right):
                result = welch_ttest(left, right)
                self.assertEqual(result["status"], status)
                self.assertEqual(
                    (result["t_statistic"], result["p_value"]), (None, None)
                )
                self.assertEqual(
                    (result["left_n"], result["right_n"]), (len(left), len(right))
                )
                if left and right:
                    self.assertAlmostEqual(result["left_mean"], statistics.mean(left))
                    self.assertAlmostEqual(result["right_mean"], statistics.mean(right))
                    self.assertAlmostEqual(
                        result["difference"],
                        statistics.mean(right) - statistics.mean(left),
                    )

    def test_welch_rejects_invalid_values(self):
        for side in range(2):
            for bad in [
                None,
                "not numeric",
                complex(1, 2),
                float("nan"),
                float("inf"),
                -float("inf"),
            ]:
                with self.subTest(side=side, bad=bad):
                    samples = [[1.0, 2.0], [3.0, 4.0]]
                    samples[side].append(bad)
                    with self.assertRaises(ValueError):
                        welch_ttest(*samples)

    def test_zero_baseline_has_no_percent_change(self):
        for left in [[0.0], [0.0, 0.0], [-1.0, 0.0, 1.0]]:
            with self.subTest(left=left):
                result = welch_ttest(left, [1.0, 2.0, 3.0])
                self.assertIsNone(result["change_percent"])
                self.assertAlmostEqual(result["difference"], 2.0)
                self.assertEqual(
                    result["status"], "insufficient_samples" if len(left) < 2 else "ok"
                )

    def test_batches_match_all_keys_without_pooling_or_mutating_files(self):
        keys = [
            ("ycsb", "run", 0, 0, "OVERALL", "Throughput", "ops/sec"),
            ("ycsb", "load", 0, 0, "OVERALL", "Throughput", "ops/sec"),
            ("ycsb", "run", 0, 1, "OVERALL", "Throughput", "ops/sec"),
            ("ycsb", "run", 1, 0, "OVERALL", "Throughput", "ops/sec"),
            ("ycsb", "run", 0, 0, "READ", "Latency", "us"),
            ("ycsb", "run", 0, 0, "UPDATE", "Latency", "us"),
            ("other-ycsb", "run", 0, 0, "OVERALL", "Throughput", "ops/sec"),
        ]
        values = ([10.0, 13.0, 20.0], [12.0, 18.0, 40.0, 70.0])
        groups = [
            [
                self.make_run(
                    f"{side}-{i}",
                    samples=[
                        dict(zip(KEYS, key), value=value + j * 1000)
                        for j, key in enumerate(keys)
                    ],
                )
                for i, value in enumerate(sample)
            ]
            for side, sample in enumerate(values)
        ]
        before = {
            file: file.read_bytes()
            for group in groups
            for run in group
            for file in run.glob("*.parquet")
        }
        for phase in ("run", "load", "all"):
            with self.subTest(phase=phase):
                report = compare_ycsb_runs(
                    groups[0], list(map(str, groups[1])), phase=phase
                )
                self.assertIsInstance(report, pl.DataFrame)
                required = (
                    set(KEYS) | STAT_FIELDS | set(COUNTS) | {"notes", "p_value_holm"}
                )
                self.assertTrue(required <= set(report.columns))
                rows = report.to_dicts()
                expected_keys = {
                    key for key in keys if phase == "all" or key[1] == phase
                }
                self.assertEqual(len(rows), len(expected_keys))
                self.assertEqual(
                    {tuple(row[key] for key in KEYS) for row in rows}, expected_keys
                )
                for row in rows:
                    offset = keys.index(tuple(row[key] for key in KEYS)) * 1000
                    expected = welch_ttest(
                        *[[value + offset for value in sample] for sample in values]
                    )
                    self.assertEqual(row["status"], "ok")
                    for field in STAT_FIELDS - {"status"}:
                        self.assertAlmostEqual(
                            row[field], expected[field], places=12, msg=field
                        )
                    self.assertEqual([row[field] for field in COUNTS], [0, 0, 0, 0])
        self.assertTrue(
            all(file.read_bytes() == content for file, content in before.items())
        )

    def test_single_collections_do_not_treat_75_metrics_as_samples(self):
        left, right = [
            self.make_run(
                side,
                samples=[
                    {"metric": f"metric-{i}", "value": float(i + offset)}
                    for i in range(75)
                ],
            )
            for side, offset in (("left", 1), ("right", 3))
        ]
        rows = compare_ycsb_runs(left, str(right)).to_dicts()
        self.assertEqual(len(rows), 75)
        for row in rows:
            self.assertEqual(
                (row["phase"], row["status"]), ("run", "insufficient_samples")
            )
            self.assertEqual((row["left_n"], row["right_n"]), (1, 1))
            self.assertAlmostEqual(row["difference"], 2.0)
            self.assertAlmostEqual(row["right_mean"], row["left_mean"] + 2.0)
            self.assertEqual(
                [row[field] for field in ("t_statistic", "p_value", "p_value_holm")],
                [None] * 3,
            )

    def test_bad_collection_blocks_testing_even_with_two_good_collections(self):
        good = [
            [self.make_run(f"{s}-{i}", float(i + s + 1)) for i in range(3)]
            for s in range(2)
        ]
        cases = [
            ([], "missing_metric"),
            ([{"metric": "Other"}], "missing_metric"),
            ([{"value": None}], "invalid_value"),
            ([{"value": float("nan")}], "invalid_value"),
            ([{"value": float("inf")}], "invalid_value"),
            ([{}, {}], "duplicate_metric"),
            ([{"execution_state": "failed"}], "invalid_invocation"),
            ([{"exit_code": 1}], "invalid_invocation"),
            ([{"exit_code": None}], "invalid_invocation"),
            ([{"parse_state": "error"}], "invalid_invocation"),
            (
                [{"argv": ["ycsb", "run", "basic", "-p", "recordcount=200"]}],
                "configuration_mismatch",
            ),
        ]
        for side in range(2):
            for samples, status in cases:
                with self.subTest(side=side, samples=samples, status=status):
                    groups = [list(group) for group in good]
                    groups[side][-1] = self.make_run("bad", samples=samples)
                    row = (
                        compare_ycsb_runs(*groups)
                        .filter(pl.col("metric") == "Throughput")
                        .row(0, named=True)
                    )
                    self.assertEqual(row["status"], status)
                    self.assertEqual(
                        (row["p_value"], row["p_value_holm"]), (None, None)
                    )
                    self.assertLessEqual(max(row["left_n"], row["right_n"]), 3)
                    affected = ("left", "right")[side]
                    self.assertEqual(
                        row[f"{affected}_missing"], int(status == "missing_metric")
                    )
                    if status == "missing_metric":
                        self.assertEqual(row[f"{affected}_n"], 2)
                        self.assertAlmostEqual(row[f"{affected}_mean"], side + 1.5)
                    if status in {"invalid_value", "invalid_invocation"}:
                        self.assertEqual(row[f"{affected}_invalid"], 1)

    def test_bad_numeric_column_is_visible(self):
        left, right = self.make_run("left"), self.make_run("right")
        file = left / "ycsb_metrics.end.parquet"
        pl.read_parquet(file).with_columns(
            pl.lit("not numeric").alias("value")
        ).write_parquet(file)
        row = compare_ycsb_runs(left, right).row(0, named=True)
        self.assertEqual(row["status"], "invalid_value")
        self.assertIsNone(row["p_value"])

    def test_units_are_separate_missing_keys_not_converted(self):
        left = self.make_run("left", 1000.0)
        right = self.make_run("right", samples=[{"value": 1.0, "unit": "ops/ms"}])
        rows = {row["unit"]: row for row in compare_ycsb_runs(left, right).to_dicts()}
        self.assertEqual(set(rows), {"ops/sec", "ops/ms"})
        self.assertEqual(
            (rows["ops/sec"]["left_mean"], rows["ops/sec"]["right_missing"]),
            (1000.0, 1),
        )
        self.assertEqual(
            (rows["ops/ms"]["right_mean"], rows["ops/ms"]["left_missing"]), (1.0, 1)
        )
        for row in rows.values():
            self.assertEqual((row["status"], row["p_value"]), ("missing_metric", None))

    def test_duplicate_paths_and_collection_ids_are_rejected(self):
        left, right = self.make_run("left"), self.make_run("right")
        clone = self.make_run("clone", cid="left")
        for lhs, rhs in [
            ([left, left], [right]),
            ([left], [right, right]),
            ([left], [left]),
            ([left, clone], [right]),
            ([left], [clone]),
        ]:
            with self.subTest(left=lhs, right=rhs), self.assertRaises(ValueError):
                compare_ycsb_runs(lhs, rhs)

    def test_malformed_inputs_raise_value_error(self):
        for table, damage in [
            ("invocations", "missing"),
            ("metrics", "missing"),
            ("invocations", "column"),
            ("metrics", "column"),
            ("invocations", "corrupt"),
            ("metrics", "corrupt"),
            ("metrics", "orphan"),
            ("invocations", "duplicate"),
            ("invocations", "empty"),
        ]:
            with self.subTest(table=table, damage=damage):
                left, right = self.make_run("left"), self.make_run("right")
                file = left / f"ycsb_{table}.end.parquet"
                if damage == "missing":
                    file.unlink()
                elif damage == "corrupt":
                    file.write_bytes(b"not a parquet file")
                else:
                    frame = pl.read_parquet(file)
                    if damage == "column":
                        frame = frame.drop("collection_id")
                    elif damage == "orphan":
                        frame = frame.with_columns(
                            pl.lit(99, dtype=pl.UInt32).alias("invocation_id")
                        )
                    elif damage == "duplicate":
                        frame = pl.concat([frame, frame])
                    else:
                        frame = frame.clear()
                        pl.DataFrame(schema=METRIC_SCHEMA).write_parquet(
                            left / "ycsb_metrics.end.parquet"
                        )
                    frame.write_parquet(file)
                with self.assertRaises(ValueError):
                    compare_ycsb_runs(left, right)

    def test_missing_metric_or_repeat_identity_is_rejected(self):
        for table, column in (
            ("invocations", "benchmark_name"),
            ("invocations", "invocation_id"),
            ("invocations", "iteration"),
            ("invocations", "phase"),
            ("metrics", "invocation_id"),
            ("metrics", "section"),
            ("metrics", "metric"),
        ):
            with self.subTest(table=table, column=column):
                left = [self.make_run(f"left-{i}", float(i)) for i in range(2)]
                right = [self.make_run(f"right-{i}", float(i + 2)) for i in range(2)]
                file = left[0] / f"ycsb_{table}.end.parquet"
                frame = pl.read_parquet(file)
                frame.with_columns(
                    pl.lit(None, dtype=frame.schema[column]).alias(column)
                ).write_parquet(file)
                with self.assertRaises(ValueError):
                    compare_ycsb_runs(left, right)

    def test_optional_outer_position_and_unit_can_be_null(self):
        groups = [
            [
                self.make_run(
                    f"{side}-{i}",
                    samples=[
                        {
                            "value": float(i + side),
                            "outer_iteration": None,
                            "unit": None,
                        }
                    ],
                )
                for i in range(2)
            ]
            for side in range(2)
        ]
        row = compare_ycsb_runs(*groups).row(0, named=True)
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["outer_iteration"])
        self.assertIsNone(row["unit"])

    def test_holm_adjusts_only_valid_tests_with_monotone_step_down(self):
        shifts = {"large": 20.0, "close_a": 5.0, "close_b": 4.8, "same": 0.0}
        values, groups = [10.0, 12.0, 14.0, 16.0], [[], []]
        for side in range(2):
            for i, value in enumerate(values):
                samples = [
                    {"metric": name, "value": value + side * shift}
                    for name, shift in shifts.items()
                ]
                samples += [
                    {"metric": "constant", "value": float(side + 1)},
                    {"metric": "invalid", "value": None if side and i == 0 else value},
                ]
                if side == 0:
                    samples.append({"metric": "missing", "value": value})
                groups[side].append(self.make_run(f"{side}-{i}", samples=samples))
        rows = compare_ycsb_runs(*groups).to_dicts()
        invalid = [row for row in rows if row["status"] != "ok"]
        self.assertEqual(
            {row["metric"]: row["status"] for row in invalid},
            {
                "constant": "zero_variance",
                "invalid": "invalid_value",
                "missing": "missing_metric",
            },
        )
        for row in invalid:
            self.assertEqual((row["p_value"], row["p_value_holm"]), (None, None))
        valid = sorted(
            (row for row in rows if row["status"] == "ok"),
            key=lambda row: row["p_value"],
        )
        self.assertEqual({row["metric"] for row in valid}, set(shifts))
        adjusted = 0.0
        for index, row in enumerate(valid):
            reference = stats.ttest_ind(
                [v + shifts[row["metric"]] for v in values], values, equal_var=False
            )
            adjusted = max(adjusted, min(1.0, (len(valid) - index) * reference.pvalue))
            self.assertAlmostEqual(row["p_value"], reference.pvalue, places=12)
            self.assertAlmostEqual(row["p_value_holm"], adjusted, places=12)

    def test_cli_csv_phase_selection_exclusive_creation_and_errors(self):
        groups = [
            [
                self.make_run(
                    f"{side}-{i}",
                    samples=[
                        {"phase": p, "value": float(i + side * 3 + 1)}
                        for p in ("load", "run")
                    ],
                )
                for i in range(2)
            ]
            for side in range(2)
        ]
        for phase in (None, "load", "all"):
            with self.subTest(phase=phase):
                output = self.root / f"report-{phase}.csv"
                command = [
                    sys.executable,
                    str(SCRIPT),
                    "--left",
                    *map(str, groups[0]),
                    "--right",
                    *map(str, groups[1]),
                    "--output",
                    str(output),
                ]
                if phase is not None:
                    command += ["--phase", phase]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = pl.read_csv(output).sort(list(KEYS))
                expected = compare_ycsb_runs(*groups, phase=phase or "run").sort(
                    list(KEYS)
                )
                self.assertEqual(
                    actual.select(*KEYS, "status").to_dicts(),
                    expected.select(*KEYS, "status").to_dicts(),
                )
                for field in (STAT_FIELDS - {"status"}) | {"p_value_holm"}:
                    for got, want in zip(actual[field], expected[field], strict=True):
                        self.assertAlmostEqual(got, want, places=12, msg=field)
                before = output.read_bytes()
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 2)
                self.assertTrue((result.stderr + result.stdout).strip())
                self.assertEqual(output.read_bytes(), before)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--left",
                str(self.root / "absent"),
                "--right",
                str(groups[1][0]),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue((result.stderr + result.stdout).strip())


if __name__ == "__main__":
    unittest.main()
