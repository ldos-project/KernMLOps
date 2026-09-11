"""Unprivileged offline integration: real Parquet/CSV/Welch, mocked original hook API."""

import csv
import fcntl
import json
import os
import statistics
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest import mock

import compare_original_matrix as comparison
import polars as pl
import run_ycsb_matrix as live_runner
import test_ycsb_matrix as fixtures


class OriginalComparisonTests(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot

        pyplot.close(
            "all"
        )  # Initialize font discovery before mocking collector subprocesses.
        self.harness = fixtures.MatrixTests()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.root = self.harness.root
        patch = mock.patch.object(comparison, "runner", fixtures.matrix)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch(
            "summarize_ycsb_matrix.summarize_matrix",
            side_effect=AssertionError("never write historical analysis"),
        )
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(
            fixtures.matrix,
            "validate_plan",
            side_effect=AssertionError(
                "never regenerate historical source/config state"
            ),
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.plans, self.events, self.directories = {}, {}, {}
        for cohort in ("historical", "candidate"):
            directory = self.root / "data/experiments" / cohort
            self.harness.directory = directory
            plan = self.harness.plan(original_only=cohort == "candidate")
            for name in ("configs", "runs"):
                (directory / name).mkdir(parents=True, exist_ok=True)
            (directory / ".lock").touch()
            for cell in plan["cells"].values():
                Path(cell["config_path"]).write_bytes(
                    fixtures.matrix.encoded(cell["config"])
                )
            self.plans[cohort] = plan
            self.events[cohort] = []
            self.directories[cohort] = directory
        self.directory = self.directories["candidate"]
        self.historical = self.directories["historical"]
        self.save_inputs()
        self.plot = comparison.plot_metrics
        patch = mock.patch.object(
            comparison,
            "plot_metrics",
            side_effect=lambda raw, overhead, coverage, category, path, phase: (
                path.write_bytes(b"test plot")
            ),
        )
        self.plot_mock = patch.start()
        self.addCleanup(patch.stop)

    def save_inputs(self):
        for cohort, directory in self.directories.items():
            (directory / "plan.json").write_text(json.dumps(self.plans[cohort]))
            (directory / "manifest.json").write_text(json.dumps({"status": "complete"}))
            (directory / "results.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in self.events[cohort])
            )

    def add_control(self):
        directory = self.root / "data/experiments/control"
        self.harness.directory = directory
        plan = self.harness.plan(selected_control_only=True)
        for name in ("configs", "runs"):
            (directory / name).mkdir(parents=True)
        (directory / ".lock").touch()
        for cell in plan["cells"].values():
            Path(cell["config_path"]).write_bytes(
                fixtures.matrix.encoded(cell["config"])
            )
        self.directories["control"] = directory
        self.plans["control"] = plan
        self.events["control"] = []
        self.save_inputs()
        return directory

    def make_run(
        self,
        cohort="candidate",
        cell="always-original-reduced",
        replicate=0,
        value=100.0,
        attempt=1,
    ):
        plan = self.plans[cohort]
        job = next(
            job
            for job in plan["jobs"]
            if job["cell"] == cell and job["replicate"] == replicate
        )
        prefix = f"{plan['experiment']}-{job['key']}-a{attempt:02d}"
        stdout = self.directories[cohort] / "runs" / f"{prefix}.stdout.log"
        stderr = stdout.with_name(f"{prefix}.stderr.log")
        stderr.touch()
        row, directory = self.harness.fixture(plan["cells"][cell], prefix, stdout)
        row.update(
            **job,
            attempt=attempt,
            status="success",
            return_code=0,
            config_path=plan["cells"][cell]["config_path"],
            stderr=str(stderr),
        )
        metrics_path = directory / "ycsb_metrics.end.parquet"
        metrics = json.loads(metrics_path.read_text())
        for metric in metrics:
            metric["value"] = value * (10 if metric["invocation_id"] == 0 else 1)
        metrics.append(
            {
                **metrics[-1],
                "metric": "AverageLatency(us)",
                "unit": "us",
                "value": value,
            }
        )
        metrics_path.write_text(json.dumps(metrics))
        fixtures.matrix.validate_result(row, plan["cells"][cell])
        for path in directory.glob("*.parquet"):
            pl.from_dicts(json.loads(path.read_text())).write_parquet(path)
        self.events[cohort].append(row)
        return row

    def compare(self, **kwargs):
        self.save_inputs()
        with mock.patch.object(
            fixtures.matrix,
            "read_rows",
            side_effect=lambda path, **options: pl.read_parquet(
                path, **options
            ).to_dicts(),
        ):
            return comparison.compare_original_matrix(
                self.directory, self.historical, **kwargs
            )

    def csv_rows(self, name, cohort="candidate"):
        with (self.directories[cohort] / "analysis" / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_complete_240_collections_exclude_all_60_archived_reduced_jobs_and_report_ten_contrasts(
        self,
    ):
        for thp in comparison.THPS:
            for mode in comparison.MODES:
                for replicate in range(30):
                    self.make_run(
                        "historical" if mode in ("none", "all") else "candidate",
                        f"{thp}-{mode}",
                        replicate,
                        100 + replicate,
                    )
        for job in self.plans["historical"]["jobs"]:
            if job["mode"] == "reduced":
                # Even successful legacy reduced rows with unusable artifacts are ignored.
                self.events["historical"].append(
                    {
                        **job,
                        "attempt": 1,
                        "status": "success",
                        "tables": {"forbidden": "/must/not/read"},
                    }
                )
        self.save_inputs()
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with mock.patch.object(
            comparison, "compare_ycsb_runs", wraps=comparison.compare_ycsb_runs
        ) as welch:
            coverage = self.compare(require_complete=True)
        self.assertEqual(
            (coverage["expected"], coverage["completed"], coverage["partial"]),
            (240, 240, False),
        )
        self.assertEqual(len(coverage["cells"]), 8)
        self.assertTrue(
            all(
                cell["completed"] == cell["expected"] == 30
                for cell in coverage["cells"].values()
            )
        )
        self.assertEqual(coverage["cohorts"]["historical"]["completed"], 120)
        self.assertEqual(coverage["cohorts"]["candidate"]["completed"], 120)
        self.assertEqual(
            coverage["cohorts"]["historical"]["excluded_modes"], ["reduced"]
        )
        self.assertEqual(welch.call_count, 10)
        self.assertEqual(len(coverage["contrasts"]), 10)
        self.assertTrue(
            all(call.kwargs == {"phase": "run"} for call in welch.call_args_list)
        )
        for contrast in coverage["contrasts"]:
            report = pl.read_csv(contrast["path"])
            self.assertEqual(set(report["phase"]), {"run"})
            self.assertEqual(set(report["left_n"]), {30})
            self.assertEqual(set(report["right_n"]), {30})
        rows = self.csv_rows("raw_metrics.csv")
        self.assertEqual({row["mode"] for row in rows}, set(comparison.MODES))
        self.assertEqual(
            {row["cohort_id"] for row in rows},
            {p["experiment"] for p in self.plans.values()},
        )
        self.assertEqual({row["phase"] for row in rows}, {"load", "run"})
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertFalse((self.historical / "analysis").exists())
        self.assertEqual(len(coverage["artifacts"]), 15)
        self.assertTrue(
            all(
                Path(path).is_relative_to(self.directory / "analysis")
                for path in coverage["artifacts"]
            )
        )

    def test_ratio_of_means_is_not_paired_and_load_run_remain_separate(self):
        for replicate, value in ((0, 100.0), (1, 300.0)):
            self.make_run("historical", "always-none", replicate, value)
        for replicate, value in ((10, 80.0), (20, 120.0)):
            self.make_run(replicate=replicate, value=value)
        coverage = self.compare()
        self.assertTrue(coverage["partial"])
        self.assertEqual(coverage["completed"], 4)
        rows = self.csv_rows("overheads.csv")
        run = next(
            row
            for row in rows
            if row["phase"] == "run" and row["metric"] == "Throughput(ops/sec)"
        )
        load = next(
            row
            for row in rows
            if row["phase"] == "load" and row["metric"] == "Throughput(ops/sec)"
        )
        self.assertEqual((run["n"], run["baseline_n"]), ("2", "2"))
        self.assertAlmostEqual(float(run["mean"]), 100)
        self.assertAlmostEqual(float(run["baseline_mean"]), 200)
        self.assertAlmostEqual(float(run["overhead_percent"]), 50)
        self.assertAlmostEqual(float(load["mean"]), 1000)
        self.assertAlmostEqual(float(load["baseline_mean"]), 2000)
        self.assertNotIn("replicate", run)
        self.assertNotIn("baseline_collection_id", run)
        runtime = next(
            row
            for row in rows
            if row["phase"] == "run" and row["metric"] == "RunTime(ms)"
        )
        self.assertAlmostEqual(float(runtime["overhead_percent"]), -50)
        self.assertIn("within each report only", coverage["notes"])
        self.assertIn("lacks the DTLB READ ACCESS", coverage["notes"])
        self.assertIn("cohort/time confounding", coverage["notes"])
        self.assertEqual(self.plot_mock.call_args.args[-1], "run")

    def test_latest_attempt_failure_or_invalid_success_never_falls_back(self):
        for cohort, cell in (
            ("candidate", "always-original-reduced"),
            ("historical", "always-none"),
        ):
            self.make_run(cohort, cell, value=10)
            newer = self.make_run(cohort, cell, value=20, attempt=2)
            self.events[cohort].append(
                {
                    **newer,
                    "attempt": 3,
                    "status": "failed",
                    "prefix": newer["prefix"].replace("a02", "a03"),
                    "stdout": newer["stdout"].replace("a02", "a03"),
                    "stderr": newer["stderr"].replace("a02", "a03"),
                }
            )
        coverage = self.compare()
        self.assertEqual(coverage["completed"], 0)
        self.assertEqual(self.csv_rows("raw_metrics.csv"), [])
        latest = self.make_run(attempt=4)
        Path(latest["tables"]["ycsb_metrics"]).unlink()
        coverage = self.compare()
        self.assertEqual(coverage["completed"], 0)
        self.assertIn(latest["key"], coverage["cohorts"]["candidate"]["invalid_jobs"])
        latest = self.make_run(attempt=5)
        coverage = self.compare()
        self.assertEqual(coverage["completed"], 1)
        self.assertEqual(
            {row["collection_id"] for row in self.csv_rows("raw_metrics.csv")},
            {latest["collection_id"]},
        )

    def test_both_matrices_require_stopped_status_and_readonly_shared_locks(self):
        for directory in self.directories.values():
            with (directory / ".lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "active"):
                    self.compare()
            for status in ("running", "ready"):
                self.save_inputs()
                (directory / "manifest.json").write_text(json.dumps({"status": status}))
                with self.assertRaisesRegex(ValueError, "stopped"):
                    comparison.compare_original_matrix(self.directory, self.historical)
        self.assertFalse((self.directory / "analysis").exists())

    def test_shared_locks_are_held_through_output_and_no_old_writes(self):
        self.make_run()

        def plot(*args):
            for directory in self.directories.values():
                with (directory / ".lock").open("rb") as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            args[4].write_bytes(b"test plot")

        with mock.patch.object(comparison, "plot_metrics", side_effect=plot):
            self.compare()
        self.assertFalse((self.historical / "analysis").exists())

    def test_system_info_and_ycsb_command_mismatch_prevent_comparison(self):
        self.make_run("historical", "always-none")
        candidate = self.make_run()
        for table, field, value, message in (
            ("system_info", "kernel_version", "different-kernel", "SystemInfo machine"),
            (
                "ycsb_invocations",
                "argv",
                ["/launcher", "load", "run", "redis", "-p", "recordcount=999"],
                "YCSB commands",
            ),
        ):
            path = Path(candidate["tables"][table])
            before = path.read_bytes()
            rows = pl.read_parquet(path).to_dicts()
            for row in rows:
                row[field] = value
            pl.from_dicts(rows).write_parquet(path)
            with self.assertRaisesRegex(ValueError, message):
                self.compare()
            self.assertFalse((self.directory / "analysis").exists())
            path.write_bytes(before)

    def test_wrong_hooks_thp_status_or_saved_table_path_invalidates_only_latest_collection(
        self,
    ):
        candidate = self.make_run()
        for table, field, value in (
            ("system_info", "hooks", ["extracted_rss_dtlb"]),
            ("system_info", "transparent_hugepages", "never"),
            ("ycsb_invocations", "parse_state", "partial"),
        ):
            path = Path(candidate["tables"][table])
            before = path.read_bytes()
            rows = pl.read_parquet(path).to_dicts()
            rows[0][field] = value
            pl.from_dicts(rows).write_parquet(path)
            coverage = self.compare()
            self.assertEqual(coverage["completed"], 0)
            self.assertIn(
                candidate["key"], coverage["cohorts"]["candidate"]["invalid_jobs"]
            )
            path.write_bytes(before)
        self.harness.original_validator.side_effect = ValueError(
            "bad final original status"
        )
        self.assertEqual(self.compare()["completed"], 0)
        self.harness.original_validator.side_effect = None
        candidate["tables"]["system_info"] = str(self.historical / "plan.json")
        self.assertEqual(self.compare()["completed"], 0)

    def test_partial_require_complete_plots_and_stale_reports(self):
        self.make_run("historical", "always-none")
        candidate = self.make_run()
        with self.assertRaisesRegex(ValueError, "partial"):
            self.compare(require_complete=True)
        self.assertFalse((self.directory / "analysis").exists())
        with mock.patch.object(comparison, "plot_metrics", side_effect=self.plot):
            coverage = self.compare()
        self.assertEqual(
            sum(c["status"] == "reported" for c in coverage["contrasts"]), 1
        )
        for name in ("overall_comparison.png", "latency_comparison.png"):
            self.assertTrue(
                (self.directory / "analysis" / name)
                .read_bytes()
                .startswith(b"\x89PNG\r\n\x1a\n")
            )
        report = next(Path(c["path"]) for c in coverage["contrasts"] if c["path"])
        self.events["candidate"].append({**candidate, "status": "invalidated"})
        self.compare()
        self.assertFalse(
            report.exists(), "a previous success report must not survive invalidation"
        )

    def test_output_cannot_escape_candidate_or_clobber_input_via_links(self):
        for output in (
            self.historical / "analysis",
            self.root / "outside",
            self.directory,
            Path("../historical/analysis"),
            Path("configs"),
            Path("runs/nested"),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.compare(output_dir=output)
        analysis = self.directory / "analysis"
        analysis.symlink_to(self.historical, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the candidate"):
            self.compare()
        analysis.unlink()
        analysis.mkdir()
        sentinel = self.historical / "plan.json"
        before = sentinel.read_bytes()
        for linked in (False, True):
            target = analysis / "raw_metrics.csv"
            if linked:
                os.link(sentinel, target)
            else:
                target.symlink_to(sentinel)
            with self.assertRaisesRegex(ValueError, "unsafe analysis output"):
                self.compare()
            self.assertEqual(sentinel.read_bytes(), before)
            target.unlink()
        coverage = self.compare(output_dir=Path("reports/original"))
        self.assertTrue(
            all(
                Path(path).is_relative_to(self.directory / "reports/original")
                for path in coverage["artifacts"]
            )
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            comparison.compare_original_matrix(self.directory, self.directory)

    def test_frozen_config_hash_and_plan_repeats_are_verified_without_current_sources(
        self,
    ):
        path = Path(self.plans["historical"]["cells"]["always-none"]["config_path"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "config hash"):
            self.compare()
        # Current source hashes differ by cohort and are not revalidated against live sources.
        cell = self.plans["historical"]["cells"]["always-none"]
        path.write_bytes(fixtures.matrix.encoded(cell["config"]))
        self.plans["historical"]["collector_sources"] = {"archived.py": "archived-sha"}
        self.compare()
        self.plans["candidate"]["repeats"] = 29
        with self.assertRaisesRegex(ValueError, "thirty repeats"):
            self.compare()

    def test_cli_forwards_explicit_output_phase_and_completion_requirement(self):
        result = {"partial": True, "completed": 0, "expected": 240, "artifacts": []}
        with mock.patch.object(
            comparison, "compare_original_matrix", return_value=result
        ) as compare:
            self.assertEqual(
                comparison.main(
                    [
                        "--directory",
                        str(self.directory),
                        "--historical",
                        str(self.historical),
                        "--output-dir",
                        "reports",
                        "--phase",
                        "all",
                        "--require-complete",
                    ]
                ),
                0,
            )
        self.assertEqual(
            compare.call_args.kwargs,
            {
                "directory": self.directory,
                "historical": self.historical,
                "output_dir": Path("reports"),
                "phase": "all",
                "require_complete": True,
            },
        )

    def test_control_complete_300_collections_keep_all_inputs_and_existing_analysis_immutable(
        self,
    ):
        from matplotlib.figure import Figure

        control = self.add_control()
        for cohort, plan in self.plans.items():
            for job in plan["jobs"]:
                if job["mode"] == "reduced":
                    self.events[cohort].append(
                        {
                            **job,
                            "status": "success",
                            "tables": {"forbidden": "/must/not/read"},
                        }
                    )
                    continue
                self.make_run(
                    cohort,
                    job["cell"],
                    job["replicate"],
                    {"historical": 100, "candidate": 120, "control": 150}[cohort]
                    + job["replicate"],
                )
        for directory in (self.historical, self.directory):
            analysis = directory / "analysis"
            analysis.mkdir()
            for name in ("coverage.json", "raw_metrics.csv", "overall_comparison.png"):
                (analysis / name).write_bytes(b"existing analysis must not change")
        self.save_inputs()
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        savefig = Figure.savefig

        def check_plot(fig, path, **kwargs):
            self.assertIn("COMPLETE 300/300", fig._suptitle.get_text())
            self.assertIn("no ITLB/flush/walk or CBMM", fig._suptitle.get_text())
            raw, overhead = fig.axes[:2]
            self.assertEqual(list(raw.get_xticks()), [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
            self.assertEqual(list(overhead.get_xticks()), [1, 2, 3, 4, 7, 8, 9, 10])
            labels = [label.get_text() for label in raw.get_xticklabels()]
            self.assertEqual(
                sum(
                    "unreduced selected" in label.replace("\n", " ")
                    for label in labels
                ),
                2,
            )
            self.assertTrue(all("n=30/30" in label for label in labels))
            self.assertEqual(
                len({tuple(points.get_facecolor()[0]) for points in raw.collections}), 5
            )
            return savefig(fig, path, **kwargs)

        with (
            mock.patch.object(
                comparison, "compare_ycsb_runs", wraps=comparison.compare_ycsb_runs
            ) as welch,
            mock.patch.object(comparison, "plot_metrics", side_effect=self.plot),
            mock.patch.object(Figure, "savefig", autospec=True, side_effect=check_plot),
        ):
            coverage = self.compare(control_directory=control, require_complete=True)
        self.assertEqual(
            (coverage["expected"], coverage["completed"], coverage["partial"]),
            (300, 300, False),
        )
        self.assertEqual(len(coverage["cells"]), 10)
        self.assertTrue(
            all(
                cell["expected"] == cell["completed"] == 30
                for cell in coverage["cells"].values()
            )
        )
        self.assertEqual(
            {cohort: info["completed"] for cohort, info in coverage["cohorts"].items()},
            {"historical": 120, "candidate": 120, "control": 60},
        )
        self.assertEqual(coverage["cohorts"]["control"]["expected"], 60)
        self.assertEqual(coverage["cohorts"]["control"]["excluded_modes"], [])
        self.assertEqual(
            coverage["cohorts"]["historical"]["excluded_modes"], ["reduced"]
        )
        self.assertEqual(welch.call_count, 18)
        self.assertEqual(len(coverage["contrasts"]), 18)
        self.assertTrue(
            all(call.kwargs == {"phase": "run"} for call in welch.call_args_list)
        )
        self.assertEqual(
            {(c["left"], c["right"]) for c in coverage["contrasts"][10:]},
            {
                (f"{thp}-selected-control", f"{thp}-{mode}")
                for thp in comparison.THPS
                for mode in (*comparison.runner.ORIGINAL_MODES, "all", "none")
            },
        )
        for contrast in coverage["contrasts"]:
            report = pl.read_csv(contrast["path"])
            self.assertEqual(set(report["phase"]), {"run"})
            self.assertEqual(set(report["left_n"]), {30})
            self.assertEqual(set(report["right_n"]), {30})
            self.assertEqual(set(report["status"]), {"ok"})
            self.assertAlmostEqual(report["left_std"][0], statistics.stdev(range(30)))
            self.assertTrue(
                all(
                    p <= h <= 1
                    for p, h in zip(report["p_value"], report["p_value_holm"])
                )
            )
        rows = self.csv_rows("raw_metrics.csv", "control")
        self.assertEqual(
            {row["mode"] for row in rows}, {*comparison.MODES, "selected-control"}
        )
        self.assertEqual({row["cohort"] for row in rows}, set(self.plans))
        self.assertEqual(
            {row["cohort_id"] for row in rows},
            {p["experiment"] for p in self.plans.values()},
        )
        self.assertEqual({row["phase"] for row in rows}, {"load", "run"})
        counts = Counter(
            (
                row["cohort_id"],
                row["thp"],
                row["mode"],
                *(row[d] for d in comparison.DIMENSIONS),
            )
            for row in rows
        )
        self.assertEqual(set(counts.values()), {30})
        for row in self.csv_rows("overheads.csv", "control"):
            self.assertEqual((row["n"], row["baseline_n"]), ("30", "30"))
            self.assertEqual(
                row["baseline_cohort_id"], self.plans["historical"]["experiment"]
            )
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        created = {
            path for path in self.root.rglob("*") if path.is_file()
        } - before.keys()
        self.assertEqual(created, {Path(path) for path in coverage["artifacts"]})
        self.assertEqual(len(created), 23)
        self.assertTrue(
            all(path.is_relative_to(control / "analysis") for path in created)
        )
        self.assertEqual(
            coverage["notes"], comparison.NOTES + " " + comparison.CONTROL_NOTES
        )
        self.harness.popen.assert_not_called()

    def test_control_means_are_unpaired_and_never_pool_phases_or_repeat_positions(self):
        control = self.add_control()
        for plan in self.plans.values():
            for cell in plan["cells"].values():
                cell["config"]["benchmark_config"]["redis"].update(
                    repeat=2, outer_repeat=2
                )
                content = fixtures.matrix.encoded(cell["config"])
                cell["config_sha256"] = fixtures.matrix.digest(content)
                Path(cell["config_path"]).write_bytes(content)
        for cohort, cell, values in (
            ("historical", "always-none", ((0, 100), (1, 300))),
            ("candidate", "always-original-reduced", ((0, 500), (1, 700))),
            ("control", "always-selected-control", ((10, 80), (20, 120))),
        ):
            for replicate, value in values:
                event = self.make_run(cohort, cell, replicate, value)
                path = Path(event["tables"]["ycsb_metrics"])
                metrics = pl.read_parquet(path).to_dicts()
                for metric in metrics:
                    metric["value"] = float(value * (metric["invocation_id"] + 1))
                pl.from_dicts(metrics).write_parquet(path)
        coverage = self.compare(control_directory=control, phase="all")
        self.assertEqual(coverage["completed"], 6)
        overheads = self.csv_rows("overheads.csv", "control")
        selected = [row for row in overheads if row["cohort"] == "control"]
        self.assertEqual(
            {
                (row["phase"], row["outer_iteration"], row["iteration"])
                for row in selected
            },
            {
                ("load", "0", "0"),
                ("load", "1", "0"),
                ("run", "0", "0"),
                ("run", "0", "1"),
                ("run", "1", "0"),
                ("run", "1", "1"),
            },
        )
        for row in selected:
            self.assertEqual((row["n"], row["baseline_n"]), ("2", "2"))
            self.assertAlmostEqual(
                float(row["mean"]) / float(row["baseline_mean"]), 0.5
            )
            self.assertAlmostEqual(
                float(row["overhead_percent"]),
                50 if row["metric"] == "Throughput(ops/sec)" else -50,
            )
            self.assertNotIn("replicate", row)
        self.assertEqual(
            {row["mean"] for row in selected if row["metric"] == "Throughput(ops/sec)"},
            {"100.0", "200.0", "300.0", "400.0", "500.0", "600.0"},
        )
        for contrast in coverage["contrasts"]:
            if contrast["path"]:
                report = pl.read_csv(contrast["path"])
                self.assertEqual(set(report["left_n"]), {2})
                self.assertEqual(set(report["right_n"]), {2})
                self.assertEqual(set(report["phase"]), {"load", "run"})

    def test_control_latest_attempts_and_invalid_telemetry_are_delegated_without_fallback(
        self,
    ):
        control = self.add_control()
        self.make_run("control", "always-selected-control", value=10)
        latest = self.make_run(
            "control", "always-selected-control", value=20, attempt=2
        )
        for status in ("failed", "running", "incomplete", "invalidated"):
            latest["status"] = status
            coverage = self.compare(control_directory=control)
            self.assertEqual(coverage["completed"], 0)
            self.assertIn(latest["key"], coverage["cohorts"]["control"]["missing_jobs"])
            self.assertEqual(self.csv_rows("raw_metrics.csv", "control"), [])
        latest["status"] = "success"
        for table, field, value in (
            ("mm_rss_stat", "pid", -1),
            ("mm_rss_stat", "member", "unknown"),
            ("dtlb_misses", "cumulative_dtlb_misses", 1 << 32),
            ("dtlb_accesses", "collection_id", "wrong-collection"),
        ):
            path = Path(latest["tables"][table][0])
            before = path.read_bytes()
            rows = pl.read_parquet(path).to_dicts()
            rows[0][field] = value
            pl.from_dicts(rows).write_parquet(path)
            with mock.patch.object(
                fixtures.matrix,
                "validate_result",
                wraps=fixtures.matrix.validate_result,
            ) as validate:
                coverage = self.compare(control_directory=control)
            validate.assert_called_once()
            self.assertEqual(validate.call_args.args[0]["attempt"], 2)
            self.assertEqual(coverage["completed"], 0)
            self.assertIn(latest["key"], coverage["cohorts"]["control"]["invalid_jobs"])
            path.write_bytes(before)
        paths = latest["tables"]["dtlb_accesses"]
        latest["tables"]["dtlb_accesses"] = latest["tables"]["dtlb_misses"]
        coverage = self.compare(control_directory=control)
        self.assertIn(
            "saved table paths",
            coverage["cohorts"]["control"]["invalid_jobs"][latest["key"]],
        )
        latest["tables"]["dtlb_accesses"] = paths
        extra = Path(latest["collection_dir"]) / "itlb_misses.end.parquet"
        pl.DataFrame({"unused": [1]}).write_parquet(extra)
        coverage = self.compare(control_directory=control)
        self.assertIn(
            "unexpected itlb_misses",
            coverage["cohorts"]["control"]["invalid_jobs"][latest["key"]],
        )
        extra.unlink()
        path = Path(paths[0])
        before = path.read_bytes()
        path.unlink()
        self.assertEqual(self.compare(control_directory=control)["completed"], 0)
        path.write_bytes(before)
        self.assertEqual(self.compare(control_directory=control)["completed"], 1)
        self.assertEqual(
            {
                row["collection_id"]
                for row in self.csv_rows("raw_metrics.csv", "control")
            },
            {latest["collection_id"]},
        )
        self.events["control"].reverse()
        with self.assertRaisesRegex(ValueError, "attempt order regressed"):
            self.compare(control_directory=control)

    def test_control_requires_all_three_stopped_and_locks_remain_held_through_output(
        self,
    ):
        control = self.add_control()
        for directory in self.directories.values():
            with (directory / ".lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "active"):
                    self.compare(control_directory=control)
            for status in ("running", "ready"):
                self.save_inputs()
                (directory / "manifest.json").write_text(json.dumps({"status": status}))
                with self.assertRaisesRegex(ValueError, "stopped"):
                    comparison.compare_original_matrix(
                        self.directory, self.historical, control_directory=control
                    )
        self.assertFalse((control / "analysis").exists())
        replace = os.replace

        def check_locks(source, target):
            for directory in self.directories.values():
                with (directory / ".lock").open("rb") as lock:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return replace(source, target)

        with mock.patch.object(
            comparison.os, "replace", side_effect=check_locks
        ) as writes:
            self.compare(control_directory=control)
        self.assertEqual(writes.call_count, 5)
        for directory in self.directories.values():
            with (directory / ".lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_control_outputs_cannot_escape_or_clobber_any_experiment_inputs(self):
        control = self.add_control()
        for output in (
            self.historical / "analysis",
            self.directory / "analysis",
            control,
            "../candidate/analysis",
            "configs",
            "runs/nested",
            self.root / "outside",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.compare(control_directory=control, output_dir=output)
        for invalid in (
            self.historical,
            self.directory,
            self.historical / "nested",
            self.directory / "nested",
            self.directory.parent,
        ):
            with (
                self.subTest(control=invalid),
                self.assertRaisesRegex(ValueError, "distinct, nonnested"),
            ):
                self.compare(control_directory=invalid)
        analysis = control / "analysis"
        analysis.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the control"):
            self.compare(control_directory=control)
        analysis.unlink()
        analysis.mkdir()
        sentinel = self.directory / "plan.json"
        before = sentinel.read_bytes()
        for name in (
            "raw_metrics.csv",
            "ttest-always-selected-control-vs-always-none.csv",
        ):
            for hardlink in (False, True):
                target = analysis / name
                if hardlink:
                    os.link(sentinel, target)
                else:
                    target.symlink_to(sentinel)
                with self.assertRaisesRegex(ValueError, "unsafe analysis output"):
                    self.compare(control_directory=control)
                self.assertEqual(sentinel.read_bytes(), before)
                target.unlink()
        cell = self.plans["control"]["cells"]["always-selected-control"]
        cell["output_dir"] = str(control / "collections")
        cell["config"]["collector_config"]["generic"]["output_dir"] = cell["output_dir"]
        content = fixtures.matrix.encoded(cell["config"])
        cell["config_sha256"] = fixtures.matrix.digest(content)
        Path(cell["config_path"]).write_bytes(content)
        event = self.make_run("control", "always-selected-control")
        for status in ("success", "failed", "invalidated"):
            event["status"] = status
            for output in (
                Path(event["collection_dir"]),
                Path(event["collection_dir"]) / "analysis",
            ):
                with (
                    self.subTest(status=status),
                    self.assertRaisesRegex(ValueError, "collection inputs"),
                ):
                    self.compare(control_directory=control, output_dir=output)
        event["status"] = "success"
        coverage = self.compare(
            control_directory=control, output_dir="reports/selected"
        )
        self.assertTrue(
            all(
                Path(path).is_relative_to(control / "reports/selected")
                for path in coverage["artifacts"]
            )
        )
        self.assertFalse((self.directory / "analysis").exists())
        self.assertFalse((self.historical / "analysis").exists())

    def test_control_time_and_row_order_may_differ_but_host_workload_and_commands_must_match(
        self,
    ):
        control = self.add_control()
        self.make_run("historical", "always-none")
        self.make_run()
        event = self.make_run("control", "always-selected-control")
        system = Path(event["tables"]["system_info"])
        pl.read_parquet(system).with_columns(
            pl.lit(999999).alias("uptime"),
            pl.lit("2026-09-09").alias("collection_time"),
        ).write_parquet(system)
        commands = Path(event["tables"]["ycsb_invocations"])
        rows = pl.read_parquet(commands).to_dicts()[::-1]
        for row in rows:
            row["argv"][0] = "/different/install/ycsb"
        pl.from_dicts(rows).write_parquet(commands)
        coverage = self.compare(control_directory=control)
        self.assertEqual(coverage["completed"], 3)
        self.assertEqual(coverage["expected"], 300)
        for table, field, value, message in (
            (system, "hostname", "different-host", "SystemInfo machine"),
            (system, "kernel_version", "different-kernel", "SystemInfo machine"),
            (
                commands,
                "argv",
                ["/ycsb", "load", "redis", "-p", "recordcount=999"],
                "YCSB commands",
            ),
        ):
            before = table.read_bytes()
            rows = pl.read_parquet(table).to_dicts()
            for row in rows:
                if table != commands or row["phase"] == "load":
                    row[field] = value
            pl.from_dicts(rows).write_parquet(table)
            with self.assertRaisesRegex(ValueError, message):
                self.compare(control_directory=control)
            table.write_bytes(before)
        cell = self.plans["control"]["cells"]["never-selected-control"]
        cell["config"]["benchmark_config"]["redis"]["record_count"] = 999
        content = fixtures.matrix.encoded(cell["config"])
        cell["config_sha256"] = fixtures.matrix.digest(content)
        Path(cell["config_path"]).write_bytes(content)
        with self.assertRaisesRegex(ValueError, "frozen Redis workload"):
            self.compare(control_directory=control)

    def test_control_frozen_plan_and_selectors_are_checked_without_regenerating_sources(
        self,
    ):
        control = self.add_control()
        plan = deepcopy(self.plans["control"])
        for field, value, message in (
            ("artifact", self.plans["candidate"]["artifact"], "reduction manifest"),
            ("repeats", 29, "thirty repeats"),
            ("seed", 18, "seed 17"),
            ("cells", self.plans["candidate"]["cells"], "unexpected matrix cells"),
            ("jobs", plan["jobs"][:-1], "invalid frozen jobs"),
            (
                "experiment",
                self.plans["candidate"]["experiment"],
                "cohort IDs must be distinct",
            ),
        ):
            self.plans["control"] = {**deepcopy(plan), field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.compare(control_directory=control)
        self.plans["control"] = deepcopy(plan)
        for cell in self.plans["control"]["cells"].values():
            cell["mode"] = "reduced"
        for job in self.plans["control"]["jobs"]:
            job["mode"] = "reduced"
        with self.assertRaisesRegex(ValueError, "cell identity mismatch"):
            self.compare(control_directory=control)
        for field, value in (
            ("hooks", ["perf", "mm_rss_stat", "madvise"]),
            ("perf_events", ["dtlb_misses", "dtlb_accesses", "itlb_misses"]),
            ("original_rss_tlb", "/not/allowed/manifest.json"),
            ("extracted_rss_tlb", "/not/allowed/manifest.json"),
            ("extracted_rss_tlb_mpmo", "/not/allowed/manifest.json"),
        ):
            self.plans["control"] = deepcopy(plan)
            cell = self.plans["control"]["cells"]["always-selected-control"]
            cell["config"]["collector_config"]["generic"][field] = value
            content = fixtures.matrix.encoded(cell["config"])
            cell["config_sha256"] = fixtures.matrix.digest(content)
            Path(cell["config_path"]).write_bytes(content)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare(control_directory=control)
        self.plans["control"] = deepcopy(plan)
        cell = self.plans["control"]["cells"]["always-selected-control"]
        Path(cell["config_path"]).write_bytes(fixtures.matrix.encoded(cell["config"]))
        cell["factory_metadata"]["manifest"] = None
        with self.assertRaisesRegex(ValueError, "control factory metadata"):
            self.compare(control_directory=control)
        self.plans["control"] = plan
        self.plans["control"]["collector_sources"] = {
            "current-perf-registry.py": "different-time-hash"
        }
        self.plans["candidate"]["collector_sources"] = {
            "old-perf-registry.py": "archived-hash"
        }
        self.assertFalse((control / "analysis").exists())
        with self.assertRaisesRegex(ValueError, "require 300 valid collections"):
            self.compare(control_directory=control, require_complete=True)
        self.compare(control_directory=control)

    def test_control_is_opt_in_and_cli_forwards_its_directory(self):
        control = self.add_control()
        with (control / ".lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            coverage = self.compare()
        self.assertEqual(coverage["expected"], 240)
        self.assertEqual(coverage["notes"], comparison.NOTES)
        self.assertEqual(set(coverage["cohorts"]), {"historical", "candidate"})
        self.assertEqual(len(coverage["contrasts"]), 10)
        self.assertTrue(
            all(
                Path(path).is_relative_to(self.directory / "analysis")
                for path in coverage["artifacts"]
            )
        )
        self.assertFalse((control / "analysis").exists())
        with mock.patch.object(
            comparison, "compare_original_matrix", return_value=coverage
        ) as compare:
            comparison.main(
                [
                    "--directory",
                    str(self.directory),
                    "--historical",
                    str(self.historical),
                    "--control-directory",
                    str(control),
                ]
            )
        self.assertEqual(
            compare.call_args.kwargs,
            {
                "directory": self.directory,
                "historical": self.historical,
                "control_directory": control,
                "output_dir": None,
                "phase": "run",
                "require_complete": False,
            },
        )


class OriginalAPIIntegrationTests(unittest.TestCase):
    """Use the real factory, manifest verifier, status validator, and table schemas."""

    def setUp(self):
        from test_original_collector import OriginalCollectorTests

        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.supplier = OriginalCollectorTests()
        # The passive manifest API hashes these bytes; these tests never dlopen them.
        self.supplier.library = self.work / "unused-queue.so"
        self.supplier.library.write_bytes(b"not loaded by runner integration")
        self.addCleanup(self.supplier.doCleanups)
        self.supplier.setUp()

    def test_real_factory_builds_120_job_plan_without_instantiation_or_native_library_load(
        self,
    ):
        from data_collection.bpf_instrumentation import original_rss_tlb as original

        environment = os.environ.copy()
        with (
            mock.patch.object(
                original.OriginalRSSDTLBHook,
                "__init__",
                side_effect=AssertionError("hook construction forbidden"),
            ),
            mock.patch.object(
                original.ct, "CDLL", side_effect=AssertionError("native load forbidden")
            ),
        ):
            plan = live_runner.make_plan(
                self.work / "matrix", 30, 17, self.supplier.path, original_only=True
            )
        self.assertEqual(len(plan["jobs"]), 120)
        for cell in plan["cells"].values():
            self.assertEqual(
                cell["factory_metadata"]["manifest"], self.supplier.manifest
            )
            self.assertEqual(cell["factory_metadata"]["hooks"], cell["expected_hooks"])
            child = live_runner.child_environment(cell)
            self.assertTrue(
                child["PYTHONPATH"].startswith(
                    str(self.supplier.bundle / "install/lib/python3/site-packages")
                )
            )
            self.assertTrue(
                child["LD_LIBRARY_PATH"].startswith(
                    str(self.supplier.bundle / "install/lib")
                )
            )
        self.assertEqual(dict(os.environ), environment)
        self.assertFalse((self.work / "matrix").exists())

    def test_real_original_tables_status_and_compiler_evidence_validate_without_bpf(
        self,
    ):
        from data_collection.bpf_instrumentation import original_rss_tlb as original

        plan = live_runner.make_plan(
            self.work / "matrix", 30, 17, self.supplier.path, original_only=True
        )
        fixture = fixtures.MatrixTests()
        fixture.root, fixture.bad_phase = self.work, False
        for mode in live_runner.ORIGINAL_MODES:
            cell = plan["cells"][f"always-{mode}"]
            cell["output_dir"] = str(self.work / "data")
            row, directory = fixture.fixture(
                cell, mode, self.work / f"{mode}.stdout.log"
            )
            status = json.loads((directory / "policy_status.end.parquet").read_text())[
                0
            ]
            hook = original.OriginalRSSDTLBHook(self.supplier.path, status["transport"])
            hook.collection_id = row["collection_id"]
            hook.evidence = Path(status["compiler_evidence"])
            hook._compiled_programs = json.loads(status["compiled_programs_json"])[
                "programs"
            ]
            hook._stats.update({name: status[name] for name in hook._stats})
            hook._rss = [((1 << 64) - 1, (1 << 64) - 1, (1 << 32) - 1, -(1 << 31))]
            hook._perf = [
                (counter, (1 << 64) - 1, (1 << 64) - 1, (1 << 32) - 1, 7, 6)
                for counter in ("dtlb_misses", "dtlb_accesses")
            ]
            hook.close()
            for path in directory.glob("*.parquet"):
                if path.name.startswith(("policy_", "original_perf")):
                    path.unlink()
                else:
                    pl.from_dicts(json.loads(path.read_text())).write_parquet(path)
            for table in hook.pop_data():
                table.table.write_parquet(directory / f"{table.name()}.end.parquet")
            live_runner.validate_result(row, cell)
            self.assertEqual(row["compiler_evidence_dir"], str(hook.evidence))
            status_path = directory / "policy_status.end.parquet"
            saved = status_path.read_bytes()
            invalid = [
                ("origin", "replacement"),
                ("callback_errors", 1),
                ("rss_received", 0),
                ("dtlb_misses_received", 0),
                ("dtlb_accesses_received", 0),
                ("rss_schema_bytes", 32),
                ("dtlb_schema_bytes", 64),
            ]
            if mode == "original-mpmo":
                invalid += [
                    ("data_slots", 31),
                    ("read_slot", 1),
                    ("polls", 0),
                    ("rss_empty", 6),
                    ("dtlb_accesses_busy", 1),
                ]
            for field, value in invalid:
                rows = pl.read_parquet(status_path).to_dicts()
                rows[0][field] = value
                pl.from_dicts(rows).write_parquet(status_path)
                with (
                    self.subTest(mode=mode, field=field),
                    self.assertRaises(ValueError),
                ):
                    live_runner.validate_result(row, cell)
                status_path.write_bytes(saved)
            if mode == "original-reduced":
                schema = pl.read_parquet_schema(directory / "original_perf.end.parquet")
                self.assertEqual(schema["cpu"], pl.UInt64)
                self.assertEqual(schema["count"], pl.UInt32)
                self.assertEqual(schema["ts_uptime_us"], pl.UInt64)
            else:
                self.assertFalse(list(directory.glob("original_perf.*.parquet")))
                self.assertFalse(list(directory.glob("policy_rss.*.parquet")))
            (hook.evidence / "rss.generated.c").write_text("tampered compiler source")
            with self.assertRaisesRegex(ValueError, "compiler evidence hash mismatch"):
                live_runner.validate_result(row, cell)


if __name__ == "__main__":
    unittest.main()
