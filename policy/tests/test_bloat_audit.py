"""Unprivileged audit tests using disposable archives, never real captures."""

import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest import mock

import audit
import polars as pl
from polars.testing import assert_frame_equal

NS = 1_000_000_000
PID = 4242
CID = "fixture-collection"
HOOKS = ["process_metadata", "process_trace", "mm_rss_stat", "memory_usage"]
REDIS_START = (
    f"{PID}:C 08 Sep 2026 00:00:00.000 # Redis version=7.4.2, bits=64, "
    f"commit=00000000, modified=0, pid={PID}, just started\n"
)


def snapshot(directory):
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


class BloatAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bloat-audit-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "KernMLOps"
        self.root.mkdir()
        patcher = mock.patch.object(audit, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.experiment = self.root / "data/experiments/fixture"
        self.directory = self.root / "data/curated/redis/fixture"
        self.output = self.root / "data/derived/bloat/fixture/audit"

    def archived(self, path):
        return str(Path("/KernMLOps") / path.relative_to(self.root))

    def write_table(self, name, frame, part="end"):
        path = self.directory / f"{name}.{part}.parquet"
        frame.write_parquet(path)
        return path

    def make_run(self):
        self.directory.mkdir(parents=True)
        (self.directory / "ycsb").mkdir()
        logs = self.experiment / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.stdout = logs / "runner.stdout.log"
        self.stderr = logs / "runner.stderr.log"
        self.stdout.write_text("Redis server v=7.4.2 bits=64\n" + REDIS_START)
        self.stderr.write_text("Lost 2 events\nDROPPED 3 samples\nother 99 events\n")
        self.event = {
            "collection_id": CID,
            "collection_dir": self.archived(self.directory),
            "stdout": self.archived(self.stdout),
            "stderr": self.archived(self.stderr),
            "thp": "always",
            "replicate": 0,
            "mode": "all",
            "status": "success",
        }
        self.frames = {
            "system_info": pl.DataFrame(
                {
                    "collection_id": [CID],
                    "benchmark_name": ["redis"],
                    "return_code": [0],
                    "transparent_hugepages": ["always"],
                    "hooks": [HOOKS[::-1]],
                }
            ),
            "process_metadata": pl.DataFrame(
                {
                    "collection_id": [CID, CID],
                    "pid": [PID, PID + 100],
                    "name": ["redis-server", "redis-server"],
                    "cmdline": ["/usr/bin/redis-server --port 6379", "redis-server"],
                }
            ),
            "process_trace": pl.DataFrame(
                {
                    "collection_id": [CID] * 5,
                    "pid": [PID, PID, PID, PID + 1, PID + 1],
                    "tgid": [PID] * 5,
                    "cap_type": ["start", "exec", "end", "start", "end"],
                    "ts_ns": [NS, NS + 1, 10 * NS, 2 * NS, 9 * NS],
                }
            ),
            "mm_rss_stat": pl.DataFrame(
                [
                    {
                        "collection_id": CID,
                        "pid": tgid + 1,
                        "tgid": tgid,
                        "ts_ns": ts,
                        "member": member,
                        "count": count,
                    }
                    for tgid, ts, member, count in (
                        (PID, 3 * NS, "MM_ANONPAGES", 40),
                        (PID, 3 * NS + 1, "MM_FILEPAGES", 5),
                        (PID, 4 * NS, "MM_ANONPAGES", 50),
                        (PID, 5 * NS, "MM_ANONPAGES", 50),
                        (PID, 7 * NS, "MM_FILEPAGES", 8),
                        (PID, 8 * NS, "MM_ANONPAGES", 45),
                        (PID, 9 * NS, "MM_ANONPAGES", 45),
                        (PID + 100, 6 * NS, "MM_ANONPAGES", 900),
                        (PID + 100, 6 * NS, "MM_SHMEMPAGES", 100),
                    )
                ]
            ),
            "ycsb_invocations": pl.DataFrame(
                [
                    {
                        "collection_id": CID,
                        "invocation_id": index,
                        "benchmark_name": "redis",
                        "phase": phase,
                        "outer_iteration": 0,
                        "iteration": 0,
                        "execution_state": "completed",
                        "exit_code": 0,
                        "parse_state": "ok",
                        "start_monotonic_ns": start * NS,
                        "end_monotonic_ns": end * NS,
                        "stderr_path": f"ycsb/{index:04d}.{phase}.stderr.log",
                    }
                    for index, (phase, start, end) in enumerate(
                        (("load", 2, 4), ("run", 5, 8))
                    )
                ]
            ),
            "ycsb_metrics": pl.DataFrame(
                {
                    "collection_id": [CID] * 5,
                    "invocation_id": [0, 1, 1, 0, 99],
                    "section": ["OVERALL", "OVERALL", "OVERALL", "READ", "OVERALL"],
                    "metric": [
                        "RunTime(ms)",
                        "RunTime(ms)",
                        "Throughput(ops/sec)",
                        "RunTime(ms)",
                        "RunTime(ms)",
                    ],
                    "value": [1500.0, 2500.0, 123.0, 999999.0, 999999.0],
                }
            ),
        }
        for name, frame in self.frames.items():
            self.write_table(name, frame)
        (self.directory / "ycsb/0000.load.stderr.log").write_text(
            "will invoke Maven to prepare the classpath\n"
            "1 sec: 100 operations\n2 sec: 200 operations\n"
        )
        (self.directory / "ycsb/0001.run.stderr.log").write_text(
            "3 sec: 300 operations\n"
        )

    def make_plan(self):
        self.experiment.mkdir(parents=True, exist_ok=True)
        (self.experiment / ".lock").touch()
        self.plan = {"repeats": 30, "all_hooks": HOOKS.copy(), "cells": {}, "jobs": []}
        for thp in ("always", "never"):
            config = self.experiment / f"{thp}-all.json"
            config.write_text(json.dumps({"hooks": HOOKS, "thp": thp}) + "\n")
            self.plan["cells"][f"{thp}-all"] = {
                "mode": "all",
                "config_path": self.archived(config),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            self.plan["jobs"].extend(
                {
                    "key": f"{thp}-all-r{replicate:02d}",
                    "cell": f"{thp}-all",
                    "mode": "all",
                    "thp": thp,
                    "replicate": replicate,
                }
                for replicate in range(30)
            )
        self.events = [
            job | {"status": "success", "attempt": 1, "collection_id": job["key"]}
            for job in self.plan["jobs"]
        ]
        self.save_plan()

    def save_plan(self, status="complete"):
        (self.experiment / "manifest.json").write_text(json.dumps({"status": status}))
        (self.experiment / "plan.json").write_text(json.dumps(self.plan))
        (self.experiment / "results.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in self.events)
        )

    def test_saved_path_rebases_container_relative_and_local_paths_at_runtime(self):
        expected = self.root / "data/curated/fixture"
        for path in (
            "/KernMLOps/data/curated/fixture",
            "data/curated/fixture",
            Path("data/curated/./other/../fixture"),
            expected,
        ):
            with self.subTest(path=path):
                self.assertEqual(audit.saved_path(path), expected)
        self.assertEqual(audit.saved_path("/KernMLOps"), self.root)
        other_root = self.root.parent / "second-workspace"
        other_root.mkdir()
        with mock.patch.object(audit, "ROOT", other_root):
            self.assertEqual(
                audit.saved_path("/KernMLOps/data/input"), other_root / "data/input"
            )
            self.assertEqual(audit.saved_path("data/input"), other_root / "data/input")
        self.assertEqual(audit.saved_path("data/input"), self.root / "data/input")

    def test_saved_path_rejects_traversal_prefix_confusion_and_symlink_escapes(self):
        outside = self.root.parent / "KernMLOps-other"
        outside.mkdir()
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        for path in (
            "../outside",
            "data/../../outside",
            "/KernMLOps/../outside",
            "/KernMLOps-other/input",
            outside / "input",
            "escape/input",
            "/KernMLOps/escape/input",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                audit.saved_path(path)
        (self.root / "inside").mkdir()
        (self.root / "alias").symlink_to(self.root / "inside", target_is_directory=True)
        self.assertEqual(audit.saved_path("alias/input"), self.root / "inside/input")

    def test_read_table_sorts_shards_relaxes_schemas_and_allows_missing_tables(self):
        self.directory.mkdir(parents=True)
        first = pl.DataFrame({"n": pl.Series([1], dtype=pl.Int32), "left": ["a"]})
        last = pl.DataFrame({"n": pl.Series([2], dtype=pl.Int64), "right": ["b"]})
        last_path = self.write_table("probe", last)
        first_path = self.write_table("probe", first, "0000")
        self.write_table("probe_other", pl.DataFrame({"n": [999]}))
        (self.directory / "probe.end.csv").write_text("not parquet\n")
        before = snapshot(self.root)
        self.assertEqual(
            audit.table_files(self.directory, "probe"), [first_path, last_path]
        )
        assert_frame_equal(
            audit.read_table(self.directory, "probe"),
            pl.DataFrame({"n": [1, 2], "left": ["a", None], "right": [None, "b"]}),
        )
        self.assertTrue(audit.read_table(self.directory, "missing").is_empty())
        self.assertEqual(snapshot(self.root), before)

    def test_audit_run_real_parquet_identity_phases_inventory_and_no_input_changes(
        self,
    ):
        self.make_run()
        rss = self.frames["mm_rss_stat"]
        first = self.write_table("mm_rss_stat", rss[:3], "0000")
        last = self.write_table("mm_rss_stat", rss[3:])
        before, event_before = snapshot(self.root), deepcopy(self.event)
        run, members, phases, inventory = audit.audit_run(self.event, HOOKS)
        self.assertEqual(run["redis_pid"], PID)
        self.assertEqual(run["collection_id"], CID)
        self.assertEqual((run["thp"], run["replicate"]), ("always", 0))
        self.assertEqual(run["collection_directory"], "data/curated/redis/fixture")
        self.assertEqual(
            (run["process_start_ns"], run["process_end_ns"]), (NS, 10 * NS)
        )
        self.assertEqual(run["rss_rows"], 7)
        self.assertEqual(run["reported_lost_events"], 5)
        self.assertFalse(run["loss_completeness_known"])
        self.assertEqual([row["phase"] for row in phases], ["load", "run"])
        self.assertEqual([row["rss_rows"] for row in phases], [3, 3])
        self.assertEqual([row["launcher_duration_s"] for row in phases], [2.0, 3.0])
        self.assertEqual(
            [row["diagnostic_reported_runtime_s"] for row in phases], [1.5, 2.5]
        )
        self.assertEqual([row["status_progress_lines"] for row in phases], [2, 1])
        self.assertEqual([row["maven_launcher"] for row in phases], [True, False])
        for row in members + phases + inventory:
            for key in ("collection_id", "thp", "replicate", "redis_pid"):
                self.assertEqual(row[key], run[key])
        tables = {row["table"]: row for row in inventory}
        self.assertEqual(list(tables), sorted(self.frames))
        for name, frame in self.frames.items():
            self.assertEqual(tables[name]["rows"], frame.height)
        self.assertEqual(
            json.loads(tables["mm_rss_stat"]["files"]),
            [str(path.relative_to(self.root)) for path in (first, last)],
        )
        self.assertEqual(
            json.loads(tables["mm_rss_stat"]["schemas"]),
            [{name: str(dtype) for name, dtype in rss.schema.items()}] * 2,
        )
        self.assertEqual(self.event, event_before)
        self.assertEqual(snapshot(self.root), before)

    def test_first_observations_are_not_backfilled_and_missing_shmem_is_not_zero(self):
        self.make_run()
        run, members, _, _ = audit.audit_run(self.event, HOOKS)
        self.assertEqual(run["rss_components"], "MM_ANONPAGES;MM_FILEPAGES")
        self.assertFalse(run["strict_resident_components_observed"])
        observed = {row["member"]: row for row in members}
        self.assertEqual(set(observed), {"MM_ANONPAGES", "MM_FILEPAGES"})
        for member, rows, minimum, maximum, levels, first, last in (
            ("MM_ANONPAGES", 5, 40, 50, 3, 3 * NS, 9 * NS),
            ("MM_FILEPAGES", 2, 5, 8, 2, 3 * NS + 1, 7 * NS),
        ):
            row = observed[member]
            self.assertEqual(
                [
                    row[key]
                    for key in (
                        "rows",
                        "min_pages",
                        "max_pages",
                        "distinct_levels",
                        "first_ns",
                        "last_ns",
                    )
                ],
                [rows, minimum, maximum, levels, first, last],
            )
            self.assertGreater(row["first_ns"], run["process_start_ns"])

    def test_observed_zero_shmem_counts_as_evidence_but_swap_does_not(self):
        self.make_run()
        rss = self.frames["mm_rss_stat"]
        for member, strict in (("MM_SWAPENTS", False), ("MM_SHMEMPAGES", True)):
            with self.subTest(member=member):
                zero = rss[:1].with_columns(
                    pl.lit(member).alias("member"),
                    pl.lit(0).alias("count"),
                    pl.lit(6 * NS).alias("ts_ns"),
                )
                self.write_table(
                    "mm_rss_stat", pl.concat([rss, zero], how="vertical_relaxed")
                )
                run, members, _, _ = audit.audit_run(self.event, HOOKS)
                self.assertEqual(run["strict_resident_components_observed"], strict)
                row = next(row for row in members if row["member"] == member)
                self.assertEqual(
                    (row["rows"], row["min_pages"], row["max_pages"]), (1, 0, 0)
                )
                self.assertEqual((row["first_ns"], row["last_ns"]), (6 * NS, 6 * NS))

    def test_relative_and_absolute_ycsb_logs_resolve_under_collection_not_repository(
        self,
    ):
        self.make_run()
        (self.root / "ycsb").mkdir()
        (self.root / "ycsb/0000.load.stderr.log").write_text("9 sec: 999 operations\n")
        local = self.directory / "ycsb/0000.load.stderr.log"
        rows = self.frames["ycsb_invocations"].to_dicts()
        for path in (
            "ycsb/0000.load.stderr.log",
            "ycsb/../ycsb/0000.load.stderr.log",
            str(local),
            self.archived(local),
        ):
            with self.subTest(path=path):
                rows[0]["stderr_path"] = path
                self.write_table("ycsb_invocations", pl.DataFrame(rows))
                before = snapshot(self.root)
                _, _, phases, _ = audit.audit_run(self.event, HOOKS)
                self.assertEqual(phases[0]["status_progress_lines"], 2)
                self.assertTrue(phases[0]["maven_launcher"])
                self.assertEqual(snapshot(self.root), before)

    def test_escaped_invocation_logs_are_rejected_without_changing_inputs(self):
        self.make_run()
        sibling = self.directory.with_name("fixture-other")
        sibling.mkdir()
        outside = sibling / "progress.log"
        outside.write_text("1 sec: 1 operations\n")
        external = self.root.parent / "progress.log"
        external.write_text("1 sec: 2 operations\n")
        (self.directory / "ycsb/link.log").symlink_to(outside)
        rows = self.frames["ycsb_invocations"].to_dicts()
        for path in (
            "../fixture-other/progress.log",
            str(outside),
            self.archived(outside),
            "ycsb/link.log",
            str(external),
        ):
            with self.subTest(path=path):
                rows[0]["stderr_path"] = path
                self.write_table("ycsb_invocations", pl.DataFrame(rows))
                before = snapshot(self.root.parent)
                with self.assertRaisesRegex(ValueError, "outside"):
                    audit.audit_run(self.event, HOOKS)
                self.assertEqual(snapshot(self.root.parent), before)

    def test_missing_and_ambiguous_explicit_redis_pids_are_rejected(self):
        self.make_run()
        for stdout in (
            "",
            "Redis server v=7.4.2 bits=64\n",
            f"redis-server pid={PID}\n",
            REDIS_START + REDIS_START.replace(f"pid={PID},", f"pid={PID + 1},"),
        ):
            with self.subTest(stdout=stdout):
                self.stdout.write_text(stdout)
                with self.assertRaisesRegex(
                    ValueError, "one explicitly logged Redis server PID"
                ):
                    audit.audit_run(self.event, HOOKS)
        self.stdout.write_text(REDIS_START * 2)
        self.assertEqual(audit.audit_run(self.event, HOOKS)[0]["redis_pid"], PID)

    def test_invalid_system_info_identity_status_thp_and_hook_multiset(self):
        self.make_run()
        info = self.frames["system_info"]
        cases = [info.clear(), pl.concat([info, info])]
        cases.extend(
            info.with_columns(pl.lit(value).alias(column))
            for column, value in (
                ("collection_id", "other"),
                ("return_code", 1),
                ("transparent_hugepages", "never"),
                ("hooks", HOOKS[:-1]),
                ("hooks", HOOKS + ["cbmm"]),
                ("hooks", HOOKS + [HOOKS[0]]),
            )
        )
        for index, frame in enumerate(cases):
            with self.subTest(case=index):
                self.write_table("system_info", frame)
                with self.assertRaisesRegex(ValueError, "invalid source SystemInfo"):
                    audit.audit_run(self.event, HOOKS)

    def test_redis_pid_requires_matching_metadata_name_and_cmdline_on_same_row(self):
        self.make_run()
        metadata = self.frames["process_metadata"]
        wrong_name = metadata.with_columns(pl.lit("python").alias("name"))
        wrong_command = metadata.with_columns(
            pl.lit("python runner.py").alias("cmdline")
        )
        cases = (
            metadata.clear(),
            metadata.filter(pl.col("pid") != PID),
            wrong_name,
            wrong_command,
            metadata.with_columns(pl.lit(None, dtype=pl.String).alias("cmdline")),
            pl.concat([wrong_name, wrong_command]),
        )
        for index, frame in enumerate(cases):
            with self.subTest(case=index):
                self.write_table("process_metadata", frame)
                with self.assertRaisesRegex(
                    ValueError, "lacks matching process metadata"
                ):
                    audit.audit_run(self.event, HOOKS)

    def test_redis_leader_requires_unique_start_and_end_in_order(self):
        self.make_run()
        trace = self.frames["process_trace"]
        leader = pl.col("pid") == PID
        start = leader & (pl.col("cap_type") == "start")
        end = leader & (pl.col("cap_type") == "end")
        cases = [
            trace.clear(),
            trace.filter(~start),
            trace.filter(~end),
            trace.filter(~leader),
            pl.concat([trace, trace.filter(start)]),
            pl.concat([trace, trace.filter(end)]),
            trace.with_columns(pl.lit(PID + 1).alias("tgid")),
        ]
        cases.extend(
            trace.with_columns(
                pl.when(end).then(timestamp).otherwise(pl.col("ts_ns")).alias("ts_ns")
            )
            for timestamp in (NS, NS - 1)
        )
        for index, frame in enumerate(cases):
            with self.subTest(case=index):
                self.write_table("process_trace", frame)
                with self.assertRaisesRegex(
                    ValueError, "ambiguous Redis leader lifetime"
                ):
                    audit.audit_run(self.event, HOOKS)

    def test_missing_or_mismatched_redis_rss_is_rejected(self):
        self.make_run()
        rss = self.frames["mm_rss_stat"]
        for frame in (
            rss.clear(),
            rss.filter(pl.col("tgid") != PID),
            rss.with_columns(pl.lit("other").alias("collection_id")),
            pl.concat(
                [rss, rss[:1].with_columns(pl.lit("other").alias("collection_id"))]
            ),
        ):
            with self.subTest(rows=frame.height):
                self.write_table("mm_rss_stat", frame)
                with self.assertRaisesRegex(
                    ValueError, "missing or mismatched Redis RSS"
                ):
                    audit.audit_run(self.event, HOOKS)

    def test_invalid_ycsb_invocation_identity_execution_exit_and_parse_status(self):
        self.make_run()
        invocations = self.frames["ycsb_invocations"]
        for column, value in (
            ("collection_id", "other"),
            ("execution_state", "running"),
            ("execution_state", "failed"),
            ("exit_code", 1),
            ("parse_state", "error"),
            ("parse_state", "partial"),
        ):
            with self.subTest(column=column, value=value):
                self.write_table(
                    "ycsb_invocations",
                    invocations.with_columns(pl.lit(value).alias(column)),
                )
                with self.assertRaisesRegex(ValueError, "invalid YCSB invocation"):
                    audit.audit_run(self.event, HOOKS)

    def test_ycsb_phases_must_fit_lifetime_with_inclusive_process_boundaries(self):
        self.make_run()
        rows = self.frames["ycsb_invocations"].to_dicts()
        for start, end in (
            (NS - 1, 4 * NS),
            (2 * NS, 10 * NS + 1),
            (3 * NS, 3 * NS),
            (4 * NS, 2 * NS),
        ):
            with self.subTest(start=start, end=end):
                rows[0].update(start_monotonic_ns=start, end_monotonic_ns=end)
                self.write_table("ycsb_invocations", pl.DataFrame(rows))
                with self.assertRaisesRegex(
                    ValueError, "outside the Redis process lifetime"
                ):
                    audit.audit_run(self.event, HOOKS)
        rows[0].update(start_monotonic_ns=NS, end_monotonic_ns=10 * NS)
        self.write_table("ycsb_invocations", pl.DataFrame(rows))
        phase = audit.audit_run(self.event, HOOKS)[2][0]
        self.assertEqual((phase["start_ns"], phase["end_ns"]), (NS, 10 * NS))
        self.assertEqual(phase["rss_rows"], 7)

    def test_missing_or_duplicate_runtime_remains_unknown_not_zero(self):
        self.make_run()
        metrics = self.frames["ycsb_metrics"]
        runtime = (pl.col("invocation_id") == 1) & (pl.col("metric") == "RunTime(ms)")
        for frame in (
            metrics.filter(~runtime),
            pl.concat([metrics, metrics.filter(runtime)]),
        ):
            with self.subTest(rows=frame.height):
                self.write_table("ycsb_metrics", frame)
                phases = audit.audit_run(self.event, HOOKS)[2]
                self.assertEqual(phases[0]["diagnostic_reported_runtime_s"], 1.5)
                self.assertIsNone(phases[1]["diagnostic_reported_runtime_s"])
                self.assertEqual(phases[1]["launcher_duration_s"], 3.0)

    def test_no_loss_messages_does_not_claim_lossless_telemetry(self):
        self.make_run()
        self.stderr.write_text("collector stopped normally\n")
        run = audit.audit_run(self.event, HOOKS)[0]
        self.assertEqual(run["reported_lost_events"], 0)
        self.assertFalse(run["loss_completeness_known"])

    def test_full_runs_selects_latest_balanced_60_sorted_without_modifying_sources(
        self,
    ):
        self.make_plan()
        retry = self.events[0] | {"attempt": 2, "collection_id": "retry-success"}
        self.events[0]["status"] = "failed"
        self.events.reverse()
        self.events.append(retry)
        for mode in ("none", "reduced"):
            job = self.plan["jobs"][0] | {"key": mode, "mode": mode, "cell": mode}
            self.plan["jobs"].append(job)
            self.plan["cells"][mode] = {"mode": mode, "config_path": "/not/read"}
            self.events.append(job | {"status": "failed", "collection_id": mode})
        self.save_plan()
        before = snapshot(self.root)
        plan, events = audit.full_runs(self.experiment)
        self.assertEqual(plan, self.plan)
        self.assertEqual(len(events), 60)
        self.assertEqual(
            Counter(event["thp"] for event in events), {"always": 30, "never": 30}
        )
        self.assertEqual({event["mode"] for event in events}, {"all"})
        self.assertEqual(events[0], retry)
        self.assertEqual(
            [(event["replicate"], event["thp"]) for event in events],
            [
                (replicate, thp)
                for replicate in range(30)
                for thp in ("always", "never")
            ],
        )
        self.assertEqual(snapshot(self.root), before)

    def test_full_runs_latest_failed_attempt_cannot_fall_back_to_success(self):
        self.make_plan()
        self.events.append(self.events[0] | {"attempt": 2, "status": "failed"})
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "sixty successful latest"):
            audit.full_runs(self.experiment)

    def test_full_runs_requires_completed_manifest_and_thirty_rounds_without_cbmm(self):
        self.make_plan()
        for status in ("running", "paused", "failed"):
            with self.subTest(status=status):
                self.save_plan(status=status)
                with self.assertRaisesRegex(ValueError, "completed, stopped"):
                    audit.full_runs(self.experiment)
        for repeats, hooks in ((29, HOOKS), (31, HOOKS), (30, HOOKS + ["cbmm"])):
            with self.subTest(repeats=repeats, hooks=hooks):
                self.plan.update(repeats=repeats, all_hooks=hooks)
                self.save_plan()
                with self.assertRaisesRegex(
                    ValueError, "thirty full-probe rounds without cbmm"
                ):
                    audit.full_runs(self.experiment)

    def test_full_runs_requires_sixty_distinct_keys_and_balanced_thp(self):
        self.make_plan()
        original = deepcopy(self.events)
        extra = self.plan["jobs"][0] | {"key": "extra-all", "replicate": 30}
        self.plan["jobs"].append(extra)
        for events in (
            original[:-1],
            original[:-1] + [original[0]],
            original + [extra | {"status": "success"}],
        ):
            with self.subTest(rows=len(events)):
                self.events = events
                self.save_plan()
                with self.assertRaisesRegex(ValueError, "sixty successful latest"):
                    audit.full_runs(self.experiment)
        self.events = original
        self.events[0]["thp"] = "never"
        self.plan["jobs"][0]["thp"] = "never"
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "unbalanced full-probe cohort"):
            audit.full_runs(self.experiment)

    def test_full_runs_rejects_results_that_differ_from_frozen_jobs(self):
        self.make_plan()
        original = self.events[0].copy()
        for column, value in (
            ("key", "unplanned"),
            ("cell", "never-all"),
            ("mode", "none"),
            ("thp", "never"),
            ("replicate", 99),
        ):
            with self.subTest(column=column):
                self.events[0] = original | {column: value}
                self.save_plan()
                with self.assertRaisesRegex(ValueError, "differs from frozen plan"):
                    audit.full_runs(self.experiment)

    def test_full_runs_checks_frozen_configuration_bytes_after_rebasing(self):
        self.make_plan()
        config = self.experiment / "never-all.json"
        self.plan["cells"]["never-all"]["config_path"] = str(
            config.relative_to(self.root)
        )
        self.save_plan()
        self.assertEqual(len(audit.full_runs(self.experiment)[1]), 60)
        config.write_bytes(config.read_bytes() + b"\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "frozen source configuration changed"):
            audit.full_runs(self.experiment)
        self.assertEqual(snapshot(self.root), before)

    def test_audit_refuses_outputs_outside_derived_bloat_before_reading_sources(self):
        derived = self.root / "data/derived/bloat"
        derived.mkdir(parents=True)
        raw = self.root / "data/curated"
        raw.mkdir()
        (derived / "escape").symlink_to(raw, target_is_directory=True)
        before = snapshot(self.root)
        with (
            mock.patch.object(audit, "full_runs") as full,
            mock.patch.object(audit, "audit_run") as per_run,
        ):
            for output in (
                self.root / "new-audit",
                self.experiment / "audit",
                self.root / "data/derived/bloat-other/audit",
                self.root.parent / "outside-audit",
                derived,
                derived / "../audit",
                derived / "escape/audit",
                "/KernMLOps/data/derived/bloat/../../curated/audit",
            ):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(ValueError, "outside|output must"),
                ):
                    audit.audit(self.experiment, output)
            full.assert_not_called()
            per_run.assert_not_called()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((raw / "audit").exists())

    def test_audit_writes_separate_kind_metadata_and_preserves_existing_derived_inputs(
        self,
    ):
        self.make_plan()
        self.make_run()
        result = audit.audit_run(self.event, HOOKS)
        self.output.parent.mkdir(parents=True)
        (self.output.parent / "manifest.json").write_text(
            '{"kind": "derived-bloat-inputs"}\n'
        )
        pl.DataFrame({"original": [42]}).write_parquet(
            self.output.parent / "runs.parquet"
        )
        before = snapshot(self.root)
        events = sorted(
            self.events, key=lambda event: (event["replicate"], event["thp"])
        )
        results = []
        for event in events:
            common = {key: event[key] for key in ("collection_id", "thp", "replicate")}
            results.append(
                (
                    result[0] | common,
                    *[[row | common for row in rows] for rows in result[1:]],
                )
            )
        printed = io.StringIO()
        with (
            mock.patch.object(audit, "audit_run", side_effect=results) as per_run,
            redirect_stdout(printed),
        ):
            report = audit.audit(
                self.archived(self.experiment), self.archived(self.output)
            )
        self.assertEqual(
            per_run.call_args_list, [mock.call(event, HOOKS) for event in events]
        )
        self.assertEqual(report["kind"], "derived-bloat-audit")
        self.assertEqual(report["source_experiment"], "data/experiments/fixture")
        self.assertEqual(
            report["source_plan_sha256"],
            hashlib.sha256((self.experiment / "plan.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (report["expected_collections"], report["audited_collections"]), (60, 60)
        )
        self.assertEqual(report["excluded"], {})
        self.assertEqual(report["by_thp"], {"always": 30, "never": 30})
        self.assertEqual(report["strict_resident_components_observed"], 0)
        self.assertEqual(
            report["component_coverage"], {"MM_ANONPAGES": 60, "MM_FILEPAGES": 60}
        )
        notes = " ".join(report["notes"])
        for note in (
            "unknown, not zero",
            "No backward fill",
            "not initial component values",
            "diagnostic only",
            "does not establish lossless",
            "no bloat labels",
        ):
            self.assertIn(note, notes)
        self.assertEqual(json.loads((self.output / "audit.json").read_text()), report)
        self.assertEqual(json.loads(printed.getvalue()), report)
        expected_files = {"audit.json"}
        for name, count in (
            ("runs", 60),
            ("rss_components", 120),
            ("phases", 120),
            ("tables", 360),
        ):
            parquet = pl.read_parquet(self.output / f"{name}.parquet")
            csv = pl.read_csv(self.output / f"{name}.csv")
            self.assertEqual(parquet.height, count)
            self.assertEqual(csv.to_dicts(), parquet.to_dicts())
            expected_files.update((f"{name}.parquet", f"{name}.csv"))
        self.assertEqual({path.name for path in self.output.iterdir()}, expected_files)
        after = snapshot(self.root)
        self.assertEqual(
            {
                path: contents
                for path, contents in after.items()
                if not path.is_relative_to(self.output.relative_to(self.root))
            },
            before,
        )

    def test_audit_refuses_nonempty_output_without_overwriting_any_inputs(self):
        self.make_plan()
        self.output.mkdir(parents=True)
        (self.output / "audit.json").write_text('{"kind": "existing-derived-data"}\n')
        pl.DataFrame({"keep": [7]}).write_parquet(self.output / "runs.parquet")
        before = snapshot(self.root)
        with (
            mock.patch.object(audit, "full_runs", return_value=(self.plan, [])),
            mock.patch.object(audit, "audit_run") as per_run,
        ):
            with self.assertRaisesRegex(
                ValueError, "derived audit output already exists"
            ):
                audit.audit(self.experiment, self.output)
            per_run.assert_not_called()
        self.assertEqual(snapshot(self.root), before)

    def test_audit_records_exclusions_without_replacing_them_with_zero_rows(self):
        self.make_plan()
        self.make_run()
        result = audit.audit_run(self.event, HOOKS)
        events = [self.event, self.event | {"collection_id": "bad-invocation"}]
        with (
            mock.patch.object(audit, "full_runs", return_value=(self.plan, events)),
            mock.patch.object(
                audit,
                "audit_run",
                side_effect=[result, ValueError("invalid YCSB invocation")],
            ),
            redirect_stdout(io.StringIO()),
        ):
            report = audit.audit(self.experiment, self.output)
        self.assertEqual(report["audited_collections"], 1)
        self.assertEqual(
            report["excluded"], {"bad-invocation": "invalid YCSB invocation"}
        )
        self.assertEqual(
            report["component_coverage"], {"MM_ANONPAGES": 1, "MM_FILEPAGES": 1}
        )
        self.assertEqual(
            pl.read_parquet(self.output / "runs.parquet")["collection_id"].to_list(),
            [CID],
        )


if __name__ == "__main__":
    unittest.main()
