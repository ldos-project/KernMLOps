import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import polars as pl
from data_schema import YCSBInvocationsTable, YCSBMetricsTable, table_types
from kernmlops_benchmark.benchmark import GenericBenchmarkConfig
from kernmlops_benchmark.errors import BenchmarkError, BenchmarkRunningError
from kernmlops_benchmark.redis import RedisBenchmark, RedisConfig


class FakeProcess:
    def __init__(self, args, pid, exit_code=0, server=False):
        self.args = args
        self.pid = pid
        self.exit_code = exit_code
        self.returncode = None
        self.server = server

    def poll(self):
        if not self.server:
            self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.exit_code
        return self.returncode

    def send_signal(self, sig):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class RedisResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.benchmark = RedisBenchmark(
            generic_config=GenericBenchmarkConfig(),
            config=RedisConfig(repeat=2, outer_repeat=2),
        )
        self.benchmark.configure_output(
            collection_id="redis-run",
            collection_dir=self.directory,
            run_label="training",
        )
        self.commands = []
        self.fail_invocation = None
        self.server = None
        self.before_spawn = None
        self.enterContext(
            patch("kernmlops_benchmark.redis.Path.exists", return_value=False)
        )
        self.enterContext(patch("kernmlops_benchmark.redis.demote", return_value=None))
        self.enterContext(
            patch(
                "kernmlops_benchmark.redis.subprocess.run",
                return_value=Mock(returncode=0),
            )
        )
        self.popen = self.enterContext(
            patch("kernmlops_benchmark.redis.subprocess.Popen", side_effect=self.spawn)
        )

    def spawn(self, args, **kwargs):
        if args[0].startswith("redis-server"):
            self.server = FakeProcess(args, 1000000, server=True)
            return self.server
        index = len(self.commands)
        self.commands.append(args)
        if self.before_spawn is not None:
            self.before_spawn()
        phase = "load" if "load" in args else "run"
        operation = "INSERT" if phase == "load" else "READ"
        kwargs["stdout"].write(
            f"[OVERALL], RunTime(ms), {100 + index}\n"
            f"[{operation}], Operations, 100\n"
            f"[{operation}], 99thPercentileLatency(us), {200 + index}\n".encode()
        )
        kwargs["stderr"].write(b"Starting test.\n")
        return FakeProcess(
            args, 1000001 + index, 7 if index == self.fail_invocation else 0
        )

    def tables(self):
        return {table.name(): table.table for table in self.benchmark.finalize()}

    def test_all_iterations_are_deferred_until_complete_benchmark(self):
        def during_cycle():
            with self.assertRaises(BenchmarkRunningError):
                self.benchmark.finalize()

        self.before_spawn = during_cycle
        with patch(
            "polars.DataFrame", side_effect=AssertionError("DataFrame during cycles")
        ):
            self.benchmark.run()
            with self.assertRaises(BenchmarkRunningError):
                self.benchmark.finalize()
            self.assertEqual(self.benchmark.poll(), 0)

        tables = self.tables()
        invocations = tables["ycsb_invocations"]
        metrics = tables["ycsb_metrics"]
        self.assertEqual(invocations["phase"].to_list(), ["load", "run", "run"] * 2)
        self.assertEqual(invocations["outer_iteration"].to_list(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(invocations["iteration"].to_list(), [0, 0, 1, 0, 0, 1])
        self.assertEqual(invocations["run_label"].to_list(), ["training"] * 6)
        self.assertEqual(invocations["execution_state"].to_list(), ["completed"] * 6)
        self.assertEqual(metrics.height, 18)
        self.assertEqual(metrics["invocation_id"].n_unique(), 6)
        self.assertEqual(len(list((self.directory / "ycsb").glob("*.log"))), 12)
        self.assertEqual(self.server.returncode, 0)
        self.assertIsNone(self.benchmark.server)
        self.assertIs(self.benchmark.finalize(), self.benchmark.finalize())

    def test_load_failure_retains_partial_output_after_cleanup(self):
        self.fail_invocation = 0
        with self.assertRaises(BenchmarkError):
            self.benchmark.run()
        with self.assertRaises(BenchmarkRunningError):
            self.benchmark.finalize()
        self.benchmark.kill()
        tables = self.tables()
        self.assertEqual(tables["ycsb_invocations"]["phase"].to_list(), ["load"])
        self.assertEqual(tables["ycsb_invocations"]["exit_code"].to_list(), [7])
        self.assertEqual(tables["ycsb_metrics"].height, 3)
        self.assertEqual(self.server.returncode, 0)

    def test_intermediate_run_failure_keeps_earlier_invocations(self):
        self.fail_invocation = 1
        with self.assertRaises(BenchmarkError):
            self.benchmark.run()
        self.benchmark.kill()
        tables = self.tables()
        self.assertEqual(tables["ycsb_invocations"]["phase"].to_list(), ["load", "run"])
        self.assertEqual(tables["ycsb_invocations"]["exit_code"].to_list(), [0, 7])
        self.assertEqual(tables["ycsb_metrics"].height, 6)

    def test_final_run_failure_is_returned_and_recorded(self):
        self.fail_invocation = 5
        self.benchmark.run()
        self.assertEqual(self.benchmark.poll(), 7)
        self.assertEqual(
            self.tables()["ycsb_invocations"]["exit_code"].to_list(), [0] * 5 + [7]
        )

    def test_wait_marks_whole_benchmark_complete(self):
        self.benchmark.run()
        self.benchmark.wait()
        self.assertEqual(self.tables()["ycsb_invocations"].height, 6)

    def test_cancel_before_run_does_not_start_server(self):
        self.benchmark.kill()
        with self.assertRaises(BenchmarkError):
            self.benchmark.run()
        self.popen.assert_not_called()
        self.assertEqual(self.tables()["ycsb_invocations"].height, 0)

    def test_cancellation_during_run_validation_can_still_finalize(self):
        benchmark = self.benchmark

        class CancelBeforeServer:
            outer_repeat = 1
            load_from_rdb = False
            tcmalloc = False

            @property
            def repeat(self):
                benchmark.kill()
                return 1

        self.benchmark.config = CancelBeforeServer()
        with self.assertRaises(BenchmarkError):
            self.benchmark.run()
        self.popen.assert_not_called()
        self.assertEqual(self.tables()["ycsb_invocations"].height, 0)

    def test_size_query_failure_does_not_prevent_server_cleanup(self):
        self.benchmark.run()
        with patch(
            "kernmlops_benchmark.redis.subprocess.run",
            side_effect=subprocess.TimeoutExpired("redis-cli", 5),
        ):
            self.assertEqual(self.benchmark.poll(), 0)
        self.assertIsNone(self.benchmark.server)

    def test_registered_tables_round_trip(self):
        self.benchmark.run()
        self.benchmark.wait()
        for table in self.benchmark.finalize():
            self.assertIn(type(table), table_types)
            path = self.directory / f"{table.name()}.end.parquet"
            table.table.write_parquet(path)
            self.assertTrue(
                table.table.equals(type(table).from_df(pl.read_parquet(path)).table)
            )
        self.assertEqual(
            {YCSBInvocationsTable.name(), YCSBMetricsTable.name()}, set(self.tables())
        )


if __name__ == "__main__":
    unittest.main()
