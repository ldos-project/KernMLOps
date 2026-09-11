"""Stdlib unittest suite; run with PYTHONPATH=python/kernmlops in make docker."""

import os
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, call, patch

import polars as pl
from data_schema.ycsb import YCSBInvocationsTable, YCSBMetricsTable
from kernmlops_benchmark import ycsb
from kernmlops_benchmark.ycsb import YCSBCapture, _parse_ycsb_line
from polars.testing import assert_frame_equal


class YCSBParserTests(unittest.TestCase):
    def test_custom_metrics_units_and_non_numeric_values(self):
        cases = [
            ("Latency(us)", "12.5", 12.5, "us"),
            ("Throughput(ops/sec)", "1.25e3", 1250.0, "ops/sec"),
            ("RunTime(ms)", "7", 7.0, "ms"),
            ("GC time(%)", "0", 0.0, "%"),
            ("Heap(MBs)", "2", 2.0, "MBs"),
            ("Operations", "3", 3.0, "count"),
            ("Return=ERROR", "1", 1.0, "count"),
            ("Custom metric", "not available", None, None),
            ("Details", "one, two, three", None, None),
            ("Status text", "[READ: Count=4, Avg=5]", None, None),
            ("Arbitrary", "-3.5", -3.5, None),
        ]
        for metric, raw, value, unit in cases:
            with self.subTest(metric=metric):
                self.assertEqual(
                    _parse_ycsb_line(f" [CUSTOM SECTION], {metric}, {raw}\n"),
                    {
                        "section": "CUSTOM SECTION",
                        "metric": metric,
                        "value_raw": raw,
                        "value": value,
                        "unit": unit,
                    },
                )

    def test_nonfinite_values_keep_raw_and_null_numeric(self):
        for raw in ("NaN", "nan", "Infinity", "+inf", "-Infinity", "1e999"):
            with self.subTest(raw=raw):
                row = _parse_ycsb_line(f"[READ], AverageLatency(us), {raw}")
                assert row is not None
                self.assertEqual(row["value_raw"], raw)
                self.assertIsNone(row["value"])

    def test_status_warnings_and_missing_measurements_are_not_zero(self):
        for line in (
            "",
            "YCSB Client 0.17.0",
            "WARNING: no operations were executed",
            "[WARN] Retry failed, retrying",
            "SLF4J: Class path contains multiple bindings.",
            "[READ: Count=4, Max=9, Min=1, Avg=5]",
            "[READ: Count=4, Max=9], [UPDATE: Count=3, Max=2]",
            "2026-09-08 00:00:01 1 sec: 4 operations; [READ: Count=4, Avg=5]",
        ):
            with self.subTest(line=line):
                self.assertIsNone(_parse_ycsb_line(line))

    def test_malformed_export_looking_lines_are_rejected(self):
        for line in (
            "[READ], Operations",
            "[READ], , 1",
            "[READ], Operations,",
            "[], Operations, 1",
            "[READ, Operations, 1",
        ):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "malformed measurement"):
                    _parse_ycsb_line(line)


class YCSBCaptureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.capture = YCSBCapture("redis")
        self.addCleanup(self.capture.cancel)

    def configure(self, **kwargs):
        self.capture.configure_output(
            collection_id="collection-1", collection_dir=self.directory, **kwargs
        )

    def spawn_output(self, stdout="", stderr="", exit_code=0, **kwargs):
        code = (
            "import sys; "
            f"sys.stdout.write({stdout!r}); sys.stderr.write({stderr!r}); "
            f"sys.exit({exit_code})"
        )
        return self.capture.spawn([sys.executable, "-c", code], **kwargs)

    def wait_for_file(self, path):
        deadline = time.monotonic() + 10
        while not path.exists() or path.stat().st_size == 0:
            if time.monotonic() >= deadline:
                self.fail(f"child did not create {path.name}")
            time.sleep(0.01)

    def test_unconfigured_and_empty_typed_tables(self):
        unused = YCSBCapture("redis")
        self.assertFalse(unused.configured)
        self.assertEqual(unused.finalize(), [])
        self.assertIs(unused.finalize(), unused.finalize())
        self.configure()
        self.assertTrue(self.capture.configured)
        self.assertEqual(list(self.directory.iterdir()), [self.directory / "ycsb"])
        self.assertEqual(list((self.directory / "ycsb").iterdir()), [])
        tables = self.capture.finalize()
        self.assertEqual(len(tables), 2)
        for table, table_type in zip(
            tables, (YCSBInvocationsTable, YCSBMetricsTable), strict=True
        ):
            self.assertIsInstance(table, table_type)
            self.assertEqual(table.table.schema, table_type.schema())
            self.assertEqual(table.table.height, 0)
            self.assertEqual(table.graphs(), [])
            self.assertIs(table.filtered_table(), table.table)

    def test_large_regular_logs_and_no_dataframe_or_parsing_between_cycles(self):
        code = "\n".join(
            [
                "import os, stat, sys",
                "assert stat.S_ISREG(os.fstat(1).st_mode)",
                "assert stat.S_ISREG(os.fstat(2).st_mode)",
                "assert os.getsid(0) == os.getpid()",
                "sys.stdout.write('progress ' + 'x' * (2 ** 21) + '\\n')",
                "sys.stderr.write('WARNING: ' + 'y' * (2 ** 21) + '\\n')",
                "print('[CUSTOM], Repeated(us), 1')",
                "print('[CUSTOM], Repeated(us), 2')",
                "print('[OTHER], Result, unknown', file=sys.stderr)",
            ]
        )
        dataframe_type = pl.DataFrame
        with (
            patch.object(
                pl, "DataFrame", side_effect=AssertionError("early DataFrame")
            ) as dataframe,
            patch.object(
                dataframe_type,
                "write_parquet",
                side_effect=AssertionError("early Parquet write"),
            ) as parquet_write,
            patch.object(
                pl, "read_parquet", side_effect=AssertionError("early Parquet read")
            ) as parquet_read,
            patch.object(
                ycsb, "_parse_ycsb_line", side_effect=AssertionError("early parsing")
            ) as parser,
            patch.object(ycsb.subprocess, "Popen", wraps=subprocess.Popen) as popen,
            patch.object(ycsb.os, "chown") as chown,
            patch.object(ycsb.os, "fchown") as fchown,
            patch.dict(os.environ, {}, clear=True),
        ):
            self.configure()
            for phase in ("load", "run"):
                process = self.capture.spawn(
                    [sys.executable, "-c", code], phase=phase, iteration=0
                )
                self.assertIsNone(process.stdout)
                self.assertIsNone(process.stderr)
                self.capture.poll(process)
                self.assertEqual(self.capture.wait(process, timeout=20), 0)
                self.assertEqual(self.capture.poll(process), 0)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertTrue(popen.call_args.kwargs["stdout"].closed)
                self.assertTrue(popen.call_args.kwargs["stderr"].closed)
            self.capture.cancel()
            dataframe.assert_not_called()
            parquet_write.assert_not_called()
            parquet_read.assert_not_called()
            parser.assert_not_called()
            chown.assert_not_called()
            fchown.assert_not_called()

        invocations, metrics = self.capture.finalize()
        self.assertEqual(invocations.table["parse_state"].to_list(), ["ok", "ok"])
        self.assertEqual(metrics.table.height, 6)
        self.assertEqual(metrics.table["value"].to_list(), [1.0, 2.0, None] * 2)
        self.assertEqual(metrics.table["line_number"].to_list(), [2, 3, 2] * 2)
        self.assertEqual(
            metrics.table["source"].to_list(), ["stdout", "stdout", "stderr"] * 2
        )
        for row in invocations.table.iter_rows(named=True):
            for source in ("stdout", "stderr"):
                path = self.directory / row[f"{source}_path"]
                self.assertTrue(stat.S_ISREG(path.stat().st_mode))
                self.assertGreater(path.stat().st_size, 2**21)
                self.assertFalse(Path(row[f"{source}_path"]).is_absolute())

    def test_phases_repeats_failures_and_labels_are_not_merged(self):
        self.configure(run_label="candidate")
        expected = [("load", 0, 0), ("run", 0, 0), ("run", 1, 0), ("load", 0, 1)]
        for index, (phase, iteration, outer) in enumerate(expected):
            process = self.spawn_output(
                stdout="[READ], Operations, 3\n[READ], Operations, 4\n",
                stderr="[CUSTOM], Value, N/A\n",
                exit_code=7 if index == 2 else 0,
                phase=phase,
                iteration=iteration,
                outer_iteration=outer,
            )
            self.assertEqual(
                self.capture.wait(process, timeout=10), 7 if index == 2 else 0
            )

        invocations, metrics = self.capture.finalize()
        rows = invocations.table.to_dicts()
        self.assertEqual(
            [(row["phase"], row["iteration"], row["outer_iteration"]) for row in rows],
            expected,
        )
        self.assertEqual([row["invocation_id"] for row in rows], list(range(4)))
        self.assertEqual(
            [row["execution_state"] for row in rows],
            ["completed", "completed", "failed", "completed"],
        )
        self.assertEqual([row["parse_state"] for row in rows], ["ok"] * 4)
        self.assertEqual([row["run_label"] for row in rows], ["candidate"] * 4)
        self.assertEqual([row["benchmark_name"] for row in rows], ["redis"] * 4)
        self.assertEqual(len({row["stdout_path"] for row in rows}), 4)
        self.assertEqual(
            metrics.table["invocation_id"].to_list(),
            [index for index in range(4) for _ in range(3)],
        )
        self.assertEqual(metrics.table["value_raw"].to_list(), ["3", "4", "N/A"] * 4)

    def test_no_metrics_and_malformed_partial_output(self):
        self.configure()
        for stdout in (
            "WARNING: empty run\n[READ: Count=1, Avg=2]\n",
            "[READ], Operations, 1\n[READ], Missing value\n[READ], Operations, 2\n",
            "[READ], Operations,\n",
        ):
            self.capture.wait(
                self.spawn_output(stdout=stdout, phase="run", iteration=0), timeout=10
            )
        invocations, metrics = self.capture.finalize()
        self.assertEqual(
            invocations.table["parse_state"].to_list(),
            ["no_metrics", "partial", "partial"],
        )
        self.assertIsNone(invocations.table["error"][0])
        self.assertIn("stdout:2: malformed measurement", invocations.table["error"][1])
        self.assertEqual(metrics.table["invocation_id"].to_list(), [1, 1])
        self.assertEqual(metrics.table["value"].to_list(), [1.0, 2.0])

    def test_missing_logs_and_parser_errors_are_distinct(self):
        self.configure()
        for iteration in range(4):
            self.capture.wait(
                self.spawn_output(
                    stdout=(
                        "[BROKEN], Value, 1\n"
                        if iteration == 3
                        else "[READ], Operations, 1\n"
                    ),
                    stderr="[CUSTOM], Value, 2\n",
                    phase="run",
                    iteration=iteration,
                ),
                timeout=10,
            )
        (self.directory / "ycsb/0000.run.stderr.log").unlink()
        (self.directory / "ycsb/0001.run.stdout.log").unlink()
        (self.directory / "ycsb/0001.run.stderr.log").unlink()

        def parse(line):
            if line.startswith("[CUSTOM]"):
                raise RuntimeError("injected parser failure")
            if line.startswith("[BROKEN]"):
                raise ValueError("unexpected parser failure")
            return _parse_ycsb_line(line)

        with patch.object(ycsb, "_parse_ycsb_line", side_effect=parse):
            invocations, metrics = self.capture.finalize()
        self.assertEqual(
            invocations.table["parse_state"].to_list(),
            ["partial", "error", "partial", "error"],
        )
        self.assertIn("stderr: missing raw log", invocations.table["error"][0])
        self.assertIn("stdout: missing raw log", invocations.table["error"][1])
        self.assertIn(
            "RuntimeError: injected parser failure", invocations.table["error"][2]
        )
        self.assertIn(
            "ValueError: unexpected parser failure", invocations.table["error"][3]
        )
        self.assertEqual(metrics.table["invocation_id"].to_list(), [0, 2])

    def test_invalid_utf8_does_not_discard_surrounding_measurements(self):
        self.configure()
        raw = b"[READ], Operations, 1\ninvalid \xff\n[READ], Operations, 2\n"
        process = self.capture.spawn(
            [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({raw!r})"],
            phase="run",
            iteration=0,
        )
        self.capture.wait(process, timeout=10)
        invocations, metrics = self.capture.finalize()
        self.assertEqual(invocations.table["parse_state"].to_list(), ["partial"])
        self.assertIn("stdout:2: decode error", invocations.table["error"][0])
        self.assertEqual(metrics.table["value"].to_list(), [1.0, 2.0])
        self.assertEqual(metrics.table["line_number"].to_list(), [1, 3])
        self.assertEqual(
            (self.directory / "ycsb/0000.run.stdout.log").read_bytes(), raw
        )

    def test_spawn_errors_survive_and_existing_logs_are_not_overwritten(self):
        self.configure()
        existing = self.directory / "ycsb/0000.load.stderr.log"
        existing.write_text("[OLD], Operations, 999\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.spawn_output(stdout="new", phase="load", iteration=0)
        self.assertEqual(existing.read_text(), "[OLD], Operations, 999\n")
        with self.assertRaises(FileNotFoundError):
            self.capture.spawn(
                [str(self.directory / "missing-program")], phase="run", iteration=1
            )
        process = self.spawn_output(
            stdout="[READ], Operations, 2\n", phase="run", iteration=2
        )
        self.capture.wait(process, timeout=10)
        invocations, metrics = self.capture.finalize()
        self.assertEqual(
            invocations.table["execution_state"].to_list(),
            ["spawn_error", "spawn_error", "completed"],
        )
        self.assertEqual(invocations.table["exit_code"].to_list(), [None, None, 0])
        self.assertIn("FileExistsError", invocations.table["error"][0])
        self.assertIn("FileNotFoundError", invocations.table["error"][1])
        self.assertEqual(metrics.table["invocation_id"].to_list(), [2])
        self.assertEqual(
            invocations.table["stdout_path"][0], "ycsb/0000.load.stdout.log"
        )
        self.assertTrue(invocations.table["end_monotonic_ns"].is_not_null().all())

    def test_invocation_recorded_before_launch_and_parent_handles_closed_on_error(self):
        self.configure()
        streams = []

        def fail_launch(*args, **kwargs):
            self.assertEqual(len(self.capture._invocations), 1)
            self.assertIsNone(self.capture._invocations[0].end_monotonic_ns)
            streams.extend((kwargs["stdout"], kwargs["stderr"]))
            raise OSError("launch failure")

        with patch.object(ycsb.subprocess, "Popen", side_effect=fail_launch):
            with self.assertRaisesRegex(OSError, "launch failure"):
                self.spawn_output(phase="load", iteration=0)
        self.assertTrue(all(stream.closed for stream in streams))
        invocations, _ = self.capture.finalize()
        self.assertEqual(
            invocations.table["execution_state"].to_list(), ["spawn_error"]
        )

    def test_owner_ids_are_explicit_and_applied_to_raw_directory_and_files(self):
        owner_ids = (os.getuid(), os.getgid())
        with (
            patch.object(ycsb.os, "chown", wraps=os.chown) as chown,
            patch.object(ycsb.os, "fchown", wraps=os.fchown) as fchown,
        ):
            self.configure(owner_ids=owner_ids)
            process = self.spawn_output(phase="load", iteration=0)
            self.capture.wait(process, timeout=10)
            chown.assert_called_once_with(self.directory / "ycsb", *owner_ids)
            self.assertEqual(fchown.call_count, 2)
            for args, _ in fchown.call_args_list:
                self.assertEqual(args[1:], owner_ids)
        for path in [self.directory / "ycsb", *(self.directory / "ycsb").iterdir()]:
            self.assertEqual((path.stat().st_uid, path.stat().st_gid), owner_ids)

    def test_preexec_fn_is_forwarded(self):
        self.configure()

        def set_umask():
            os.umask(0o077)

        process = self.capture.spawn(
            [sys.executable, "-c", "import os; assert os.umask(0) == 0o077"],
            phase="load",
            iteration=0,
            preexec_fn=set_umask,
        )
        self.assertEqual(self.capture.wait(process, timeout=10), 0)

    def test_end_timestamp_is_first_observation_and_finalize_is_idempotent(self):
        self.configure()
        argv = [sys.executable, "-c", "print('[READ], Operations, 1')"]
        original_argv = list(argv)
        with patch.object(ycsb.time, "monotonic_ns", return_value=100):
            process = self.capture.spawn(argv, phase="run", iteration=0)
        argv.append("not part of invocation")
        with patch.object(ycsb.time, "monotonic_ns", return_value=200):
            self.capture.wait(process, timeout=10)
        with (
            patch.object(ycsb.time, "monotonic_ns", return_value=300),
            patch.object(ycsb, "_parse_ycsb_line", wraps=_parse_ycsb_line) as parser,
        ):
            self.assertEqual(self.capture.poll(process), 0)
            tables = self.capture.finalize()
            calls = parser.call_count
            self.assertGreater(calls, 0)
            self.assertIs(self.capture.finalize(), tables)
            self.assertEqual(parser.call_count, calls)
        row = tables[0].table.row(0, named=True)
        self.assertEqual(row["start_monotonic_ns"], 100)
        self.assertEqual(row["end_monotonic_ns"], 200)
        self.assertEqual(row["argv"], original_argv)
        self.assertIsNone(row["run_label"])
        self.assertIsNone(row["outer_iteration"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.spawn_output(phase="run", iteration=1)

    def test_active_child_rejection_timeout_and_cancel_while_waiting(self):
        self.configure()
        ready = self.directory / "ready"
        code = (
            "import pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('[READ], Operations, 1', flush=True); "
            f"pathlib.Path({str(ready)!r}).write_text('ready'); "
            "time.sleep(60)"
        )
        process = self.capture.spawn(
            [sys.executable, "-c", code], phase="run", iteration=0
        )
        self.wait_for_file(ready)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.capture.wait(process, timeout=0.01)
        with (
            patch.object(ycsb, "_parse_ycsb_line") as parser,
            patch.object(pl, "DataFrame") as dataframe,
        ):
            with self.assertRaisesRegex(RuntimeError, "active"):
                self.capture.finalize()
            parser.assert_not_called()
            dataframe.assert_not_called()
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(self.capture.wait, process, 10)
            with (
                patch.object(ycsb, "_TERM_TIMEOUT", 0.05),
                patch.object(
                    pl, "DataFrame", side_effect=AssertionError("early frame")
                ) as dataframe,
                patch.object(ycsb, "_parse_ycsb_line") as parser,
            ):
                self.capture.cancel()
                parser.assert_not_called()
                dataframe.assert_not_called()
            self.assertEqual(waiting.result(timeout=10), -signal.SIGKILL)
        self.capture.cancel()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.spawn_output(phase="run", iteration=1)
        invocations, metrics = self.capture.finalize()
        self.assertEqual(invocations.table["execution_state"].to_list(), ["cancelled"])
        self.assertEqual(invocations.table["exit_code"].to_list(), [-signal.SIGKILL])
        self.assertEqual(metrics.table["value"].to_list(), [1.0])

    def test_cancel_does_not_signal_completed_or_externally_reaped_children(self):
        self.configure()
        observed = self.spawn_output(phase="load", iteration=0)
        self.capture.wait(observed, timeout=10)
        unobserved = self.spawn_output(phase="run", iteration=0)
        unobserved.wait(timeout=10)
        with patch.object(ycsb.os, "killpg") as killpg:
            self.capture.cancel()
            killpg.assert_not_called()
        invocations, _ = self.capture.finalize()
        self.assertEqual(
            invocations.table["execution_state"].to_list(), ["completed"] * 2
        )

    def test_wait_stops_if_cancel_cannot_reap_client(self):
        self.configure()
        child = Mock(pid=1000000000, args=["stub"], returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("stub", 0)
        try:
            with (
                patch.object(ycsb.subprocess, "Popen", return_value=child),
                patch.object(ycsb.os, "killpg"),
                patch.object(ycsb, "_TERM_TIMEOUT", 0),
            ):
                self.capture.spawn(["stub"], phase="load", iteration=0)
                with self.assertRaisesRegex(RuntimeError, "did not exit"):
                    self.capture.cancel()
                with self.assertRaisesRegex(RuntimeError, "active after cancellation"):
                    self.capture.wait(child)
                with self.assertRaisesRegex(RuntimeError, "active"):
                    self.capture.finalize()
        finally:
            child.poll.return_value = -signal.SIGKILL
            child.returncode = -signal.SIGKILL

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Linux pidfd support required")
    def test_cancel_kills_resistant_descendant_after_group_leader_exits(self):
        self.configure()
        ready = self.directory / "descendant"
        worker = (
            "import os, pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        leader = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {worker!r}]); "
            "time.sleep(60)"
        )
        process = self.capture.spawn(
            [sys.executable, "-c", leader], phase="run", iteration=0
        )
        self.wait_for_file(ready)
        # A pidfd observes exit without /proc reads or signalling a reused PID.
        pidfd = os.pidfd_open(int(ready.read_text()))
        try:
            with (
                patch.object(ycsb, "_TERM_TIMEOUT", 0.1),
                patch.object(ycsb.os, "killpg", wraps=os.killpg) as killpg,
            ):
                self.capture.cancel()
            self.assertEqual(self.capture.poll(process), -signal.SIGTERM)
            self.assertEqual(
                killpg.call_args_list,
                [call(process.pid, signal.SIGTERM), call(process.pid, signal.SIGKILL)],
            )
            self.assertTrue(select.select([pidfd], [], [], 10)[0])
            with self.assertRaises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)
        finally:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.close(pidfd)

    def test_spawn_and_cancel_are_serialized(self):
        self.configure()
        entering_launch = threading.Event()
        release_launch = threading.Event()
        cancelling = threading.Event()
        real_popen = subprocess.Popen

        def delayed_launch(*args, **kwargs):
            entering_launch.set()
            if not release_launch.wait(timeout=10):
                raise RuntimeError("test did not release launch")
            return real_popen(*args, **kwargs)

        def cancel():
            cancelling.set()
            self.capture.cancel()

        with (
            patch.object(ycsb.subprocess, "Popen", side_effect=delayed_launch),
            patch.object(ycsb, "_TERM_TIMEOUT", 0.05),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            launching = pool.submit(
                self.capture.spawn,
                [sys.executable, "-c", "import time; time.sleep(60)"],
                phase="run",
                iteration=0,
            )
            try:
                self.assertTrue(entering_launch.wait(timeout=10))
                cancelled = pool.submit(cancel)
                self.assertTrue(cancelling.wait(timeout=10))
            finally:
                release_launch.set()
            process = launching.result(timeout=10)
            cancelled.result(timeout=10)
        self.assertIsNotNone(self.capture.poll(process))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.spawn_output(phase="run", iteration=1)

    def test_parquet_roundtrip_only_after_finalize(self):
        self.configure()
        process = self.spawn_output(
            stdout="[READ], Latency(us), NaN\n[READ], Latency(us), 2.5\n",
            phase="run",
            iteration=0,
        )
        self.capture.wait(process, timeout=10)
        tables = self.capture.finalize()
        for table in tables:
            path = self.directory / f"{table.name()}.parquet"
            table.table.write_parquet(path)
            restored = type(table).from_df(pl.read_parquet(path))
            self.assertEqual(restored.table.schema, type(table).schema())
            assert_frame_equal(restored.table, table.table)
            self.assertEqual(restored.graphs(), [])


if __name__ == "__main__":
    unittest.main()
