import contextlib
import csv
import fcntl
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[1] / "summarize_ycsb_matrix.py"
SPEC = importlib.util.spec_from_file_location("ycsb_matrix_summary_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
THROUGHPUT = "Throughput(ops/sec)"
INVOCATIONS = {
    "collection_id": pl.String,
    "invocation_id": pl.UInt32,
    "benchmark_name": pl.String,
    "phase": pl.String,
    "outer_iteration": pl.UInt32,
    "iteration": pl.UInt32,
    "execution_state": pl.String,
    "exit_code": pl.Int32,
    "parse_state": pl.String,
}
METRICS = {
    "collection_id": pl.String,
    "invocation_id": pl.UInt32,
    "section": pl.String,
    "metric": pl.String,
    "unit": pl.String,
    "value": pl.Float64,
}


class YCSBMatrixSummaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / ".lock").touch()
        self.plan = {"repeats": 30, "all_hooks": ["rss_stat"], "cells": {}, "jobs": []}
        self.events, self.status = [], "paused"
        for thp in MODULE.THPS:
            for mode in MODULE.MODES:
                name = f"{thp}-{mode}"
                hooks = self.plan["all_hooks"].copy() if mode == "all" else []
                self.plan["cells"][name] = {
                    "thp": thp,
                    "mode": mode,
                    "expected_hooks": ["extracted_rss_dtlb"]
                    if mode == "reduced"
                    else hooks.copy(),
                    "config": {
                        "collector_config": {
                            "generic": {
                                "hooks": hooks,
                                "extracted_rss_tlb": "/frozen/manifest.json"
                                if mode == "reduced"
                                else None,
                            }
                        },
                        "benchmark_config": {
                            "generic": {
                                "benchmark": "redis",
                                "transparent_hugepages": thp,
                            },
                            "redis": {"repeat": 2, "outer_repeat": 2},
                        },
                    },
                }
                self.plan["jobs"] += [
                    {
                        "key": f"{name}-r{r:02d}",
                        "cell": name,
                        "replicate": r,
                        "thp": thp,
                        "mode": mode,
                    }
                    for r in range(30)
                ]

    def save_inputs(self):
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        (self.root / "manifest.json").write_text(json.dumps({"status": self.status}))
        if self.events:
            (self.root / "results.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in self.events)
            )

    def summarize(self):
        self.save_inputs()
        return MODULE.summarize_matrix(self.root)

    def make_run(
        self, cell="always-none", replicate=0, value=100.0, *, attempt=1, samples=None
    ):
        job = next(
            j
            for j in self.plan["jobs"]
            if j["cell"] == cell and j["replicate"] == replicate
        )
        cid = f"{job['key']}-a{attempt}"
        directory = self.root / "captures" / cid
        directory.mkdir(parents=True)
        positions = [("load", o, 0) for o in range(2)] + [
            ("run", o, i) for o in range(2) for i in range(2)
        ]
        invocations = [
            {
                "collection_id": cid,
                "invocation_id": n,
                "benchmark_name": "redis",
                "phase": p,
                "outer_iteration": o,
                "iteration": i,
                "execution_state": "completed",
                "exit_code": 0,
                "parse_state": "ok",
            }
            for n, (p, o, i) in enumerate(positions)
        ]
        metrics = []
        for sample in [{"value": value}] if samples is None else samples:
            position = (
                sample.get("phase", "run"),
                sample.get("outer_iteration", 0),
                sample.get("iteration", 0),
            )
            metrics.append(
                {
                    "collection_id": cid,
                    "invocation_id": positions.index(position),
                    "section": sample.get("section", "OVERALL"),
                    "metric": sample.get("metric", THROUGHPUT),
                    "unit": sample.get("unit", "ops/sec"),
                    "value": sample["value"],
                }
            )
        tables = {
            name: str(directory / f"{name}.end.parquet")
            for name in ("system_info", "ycsb_invocations", "ycsb_metrics")
        }
        pl.DataFrame(invocations, schema=INVOCATIONS).write_parquet(
            tables["ycsb_invocations"]
        )
        pl.DataFrame(metrics, schema=METRICS).write_parquet(tables["ycsb_metrics"])
        pl.DataFrame(
            [
                {
                    "collection_id": cid,
                    "benchmark_name": "redis",
                    "return_code": 0,
                    "hooks": self.plan["cells"][cell]["expected_hooks"],
                }
            ],
            schema={
                "collection_id": pl.String,
                "benchmark_name": pl.String,
                "return_code": pl.Int32,
                "hooks": pl.List(pl.String),
            },
        ).write_parquet(tables["system_info"])
        if job["mode"] == "reduced":
            for name, column in (
                ("policy_rss", "count_pages"),
                ("policy_dtlb", "dtlb_missrate"),
            ):
                path = directory / f"{name}.end.parquet"
                pl.DataFrame({column: [1.0]}).write_parquet(path)
                tables[name] = [str(path)]
            tables["policy_status"] = []
        event = {
            **job,
            "attempt": attempt,
            "status": "success",
            "return_code": 0,
            "collection_id": cid,
            "collection_dir": str(directory),
            "tables": tables,
        }
        self.events.append(event)
        return event

    def csv_rows(self, name):
        with (self.root / "analysis" / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_matched_rounds_signs_all_metrics_and_no_capture_changes(self):
        for r, base, mode in ((0, 100.0, 80.0), (1, 200.0, 220.0)):
            self.make_run(replicate=r, value=base)
            self.make_run("always-all", r, mode)
        self.make_run("never-none", value=400.0)
        self.make_run("never-all", value=200.0)
        self.make_run("always-reduced", value=110.0)
        metrics = [
            ("OVERALL", "RunTime(ms)", "ms"),
            *[
                ("READ", f"{name}Latency(us)", "us")
                for name in ("Average", "95thPercentile", "99thPercentile")
            ],
            ("READ", "Operations", None),
            ("READ", "Return=ERROR", None),
            ("READ", "17", None),
        ]
        for cell, value in (
            ("always-none", 10.0),
            ("always-all", 12.0),
            ("always-reduced", 5.0),
        ):
            self.make_run(
                cell,
                2,
                samples=[
                    {"section": s, "metric": m, "unit": u, "value": value}
                    for s, m, u in metrics
                ],
            )
        before = {
            p: p.read_bytes() for p in (self.root / "captures").rglob("*.parquet")
        }
        coverage = self.summarize()
        changes = {
            (r["thp"], r["mode"], int(r["replicate"]), r["metric"]): float(
                r["overhead_percent"]
            )
            for r in self.csv_rows("overheads.csv")
        }
        for key, expected in (
            (("always", "all", 0, THROUGHPUT), 20),
            (("always", "all", 1, THROUGHPUT), -10),
            (("never", "all", 0, THROUGHPUT), 50),
            (("always", "reduced", 0, THROUGHPUT), -10),
        ):
            self.assertAlmostEqual(changes[key], expected)
        for _, metric, _ in metrics[:4]:
            self.assertAlmostEqual(changes["always", "all", 2, metric], 20)
            self.assertAlmostEqual(changes["always", "reduced", 2, metric], -50)
        self.assertEqual(len(changes), 12)
        self.assertTrue(
            {m for _, m, _ in metrics}
            <= {r["metric"] for r in self.csv_rows("raw_metrics.csv")}
        )
        self.assertEqual(
            (coverage["completed"], coverage["expected"], coverage["partial"]),
            (10, 180, True),
        )
        self.assertEqual(
            coverage["cells"]["always-none"], {"expected": 30, "completed": 3}
        )
        self.assertTrue(
            all(p.read_bytes() == contents for p, contents in before.items())
        )
        self.assertTrue(
            all(
                Path(p).is_file() and Path(p).stat().st_size
                for p in coverage["artifacts"]
            )
        )

    def test_phase_outer_inner_and_units_are_not_cross_paired(self):
        base = [
            {"phase": "load", "value": 100.0},
            {"outer_iteration": 1, "value": 200.0},
            {"iteration": 1, "value": 300.0},
            {"unit": "ops/ms", "value": 400.0},
        ]
        self.make_run(samples=base)
        mode = self.make_run(
            "always-all",
            samples=[
                {"value": 50.0},
                {"outer_iteration": 1, "iteration": 1, "value": 60.0},
                {"phase": "load", "value": 80.0},
            ],
        )
        coverage = self.summarize()
        rows = self.csv_rows("overheads.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (rows[0]["phase"], rows[0]["outer_iteration"], rows[0]["iteration"]),
            ("load", "0", "0"),
        )
        self.assertAlmostEqual(float(rows[0]["overhead_percent"]), 20)
        self.assertEqual(coverage["excluded_overhead_rows"], 2)
        # Invocation IDs alone are insufficient: a foreign CID must invalidate the collection.
        path = mode["tables"]["ycsb_metrics"]
        pl.read_parquet(path).with_columns(
            pl.lit("foreign").alias("collection_id")
        ).write_parquet(path)
        self.assertIn(mode["key"], self.summarize()["invalid_jobs"])

    def test_missing_zero_negative_and_nonfinite_baselines_are_excluded(self):
        for r, value in enumerate((0.0, -1.0, float("nan"), float("inf"), None)):
            self.make_run(replicate=r, value=value)
            self.make_run("always-all", r, 10.0)
        self.make_run("always-all", 5, 10.0)  # No baseline in this round.
        self.make_run(replicate=6)
        self.make_run("always-all", 6, float("nan"))
        self.make_run(replicate=7, samples=[])
        self.make_run("always-reduced", 7)
        coverage = self.summarize()
        self.assertEqual(self.csv_rows("overheads.csv"), [])
        self.assertEqual(coverage["excluded_overhead_rows"], 8)
        self.assertEqual(len(self.csv_rows("raw_metrics.csv")), 14)

    def test_latest_success_only_and_invalid_artifacts_do_not_fall_back(self):
        self.status = "complete"  # The manifest alone must not imply full coverage.
        old = self.make_run(value=10.0)
        old_path = Path(old["tables"]["ycsb_metrics"])
        old_bytes = old_path.read_bytes()
        old_path.unlink()
        latest = self.make_run(value=30.0, attempt=2)
        self.events.append({**latest, "attempt": 3, "status": "failed"})
        failed = next(j for j in self.plan["jobs"] if j["cell"] == "never-all")
        self.events.append({**failed, "attempt": 1, "status": "failed"})
        coverage = self.summarize()
        self.assertEqual(coverage["completed"], 0)
        self.assertTrue(coverage["partial"])
        self.assertEqual(self.csv_rows("raw_metrics.csv"), [])
        self.assertIn(latest["key"], coverage["missing_jobs"])
        latest = self.make_run(value=40.0, attempt=4)
        coverage = self.summarize()
        self.assertEqual(coverage["completed"], 1)
        self.assertEqual(
            [float(r["value"]) for r in self.csv_rows("raw_metrics.csv")], [40.0]
        )
        self.assertEqual(
            self.csv_rows("raw_metrics.csv")[0]["collection_id"],
            latest["collection_id"],
        )
        self.assertIn(failed["key"], coverage["missing_jobs"])
        old_path.write_bytes(old_bytes)
        latest["tables"]["extra"] = [
            str(Path(latest["collection_dir"]) / "missing.parquet")
        ]
        coverage = self.summarize()
        self.assertEqual(coverage["completed"], 0)
        self.assertIn(latest["key"], coverage["invalid_jobs"])

    def test_cbmm_is_rejected_everywhere_in_plan(self):
        cell = self.plan["cells"]["always-all"]
        for hooks in (
            self.plan["all_hooks"],
            cell["expected_hooks"],
            cell["config"]["collector_config"]["generic"]["hooks"],
        ):
            hooks.append("cbmm")
            with self.assertRaisesRegex(ValueError, "cbmm"):
                self.summarize()
            hooks.pop()
        self.assertFalse((self.root / "analysis").exists())

    def test_running_or_ready_manifest_and_active_lock_are_rejected(self):
        for status in ("running", "ready"):
            self.status = status
            with self.assertRaisesRegex(ValueError, "stopped"):
                self.summarize()
        self.status = "paused"
        with (self.root / ".lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active"):
                self.summarize()
        self.assertFalse((self.root / "analysis").exists())

    def test_prepared_empty_matrix_cli_has_headers_counts_and_five_artifacts(self):
        self.save_inputs()
        self.assertFalse((self.root / "results.jsonl").exists())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(MODULE.main(["--directory", str(self.root)]), 0)
        self.assertIn("PARTIAL: 0/180", output.getvalue())
        coverage = json.loads((self.root / "analysis" / "coverage.json").read_text())
        self.assertEqual(len(coverage["artifacts"]), 5)
        self.assertEqual(len(coverage["missing_jobs"]), 180)
        self.assertEqual(self.csv_rows("raw_metrics.csv"), [])
        self.assertEqual(self.csv_rows("overheads.csv"), [])
        self.assertTrue(
            all(
                c == {"completed": 0, "expected": 30}
                for c in coverage["cells"].values()
            )
        )
        for name in ("overall_overhead.png", "latency_overhead.png"):
            self.assertTrue(
                (self.root / "analysis" / name)
                .read_bytes()
                .startswith(b"\x89PNG\r\n\x1a\n")
            )


if __name__ == "__main__":
    unittest.main()
