import signal
import unittest
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli as cli_module
import polars as pl
from cli import collect
from data_schema import CollectionTable
from kernmlops_benchmark.benchmark import (
    Benchmark,
    FauxBenchmark,
    FauxBenchmarkConfig,
    GenericBenchmarkConfig,
)


class FakeMetricsTable(CollectionTable):
    table_name = "test_ycsb_metrics"

    @classmethod
    def name(cls):
        return cls.table_name

    def __init__(self, table):
        self._table = table

    @property
    def table(self):
        return self._table


class FakeRunsTable(FakeMetricsTable):
    table_name = "test_ycsb_runs"


class FakeHook:
    def __init__(self, events, name="fake_hook"):
        self.events = events
        self.hook_name = name
        self.loaded = False
        self.closed = False
        self.load_error = None
        self.close_error = None
        self.pop_error = None

    def name(self):
        return self.hook_name

    def load(self, collection_id):
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True
        self.events.append(f"{self.name()}:load")

    def poll(self):
        if self.closed:
            raise AssertionError("Polled a closed hook")

    def close(self):
        self.closed = True
        self.events.append(f"{self.name()}:close")
        if self.close_error is not None:
            raise self.close_error

    def pop_data(self):
        self.events.append(f"{self.name()}:pop")
        if self.pop_error is not None:
            raise self.pop_error
        return []


class FakeBenchmark(Benchmark):
    def __init__(self, events, before_finalize):
        self.events = events
        self.before_finalize = before_finalize
        self.output = None
        self.cycles = []
        self.started = False
        self.finished = Event()
        self.cancelled = Event()
        self.clients_stopped = True
        self.return_code = 0
        self.run_error = None
        self.poll_error = None
        self.kill_error = None
        self.finalize_error = None
        self.after_cycle = None
        self.kill_count = 0
        self.finalize_count = 0

    @classmethod
    def name(cls):
        return "redis"

    def is_configured(self):
        return True

    def setup(self):
        self.events.append("setup")

    def configure_output(self, **kwargs):
        self.output = kwargs
        if not kwargs["collection_dir"].is_dir():
            raise AssertionError("Output directory must exist before configuration")
        self.events.append("configure")

    def run(self):
        if self.output is None:
            raise AssertionError("Output must be configured before run")
        self.events.append("run")
        self.clients_stopped = False
        self.started = True
        for repeat in range(2):
            for phase in ("load", "run"):
                if self.cancelled.is_set():
                    return
                self.cycles.append((repeat, phase))
                self.events.append(f"cycle:{repeat}:{phase}")
                if self.run_error is not None:
                    raise self.run_error
                if self.after_cycle is not None:
                    self.after_cycle()
        self.clients_stopped = True
        self.finished.set()

    def poll(self):
        if not self.started:
            return None
        if self.poll_error is not None:
            raise self.poll_error
        return self.return_code if self.finished.is_set() else None

    def kill(self):
        self.kill_count += 1
        self.clients_stopped = True
        self.cancelled.set()
        self.events.append("kill")
        if self.kill_error is not None:
            raise self.kill_error

    def finalize(self):
        self.finalize_count += 1
        self.before_finalize()
        self.events.append("finalize")
        if self.finalize_error is not None:
            raise self.finalize_error
        # All fake YCSB frames are deliberately created here, not during cycles.
        metrics = pl.DataFrame(
            {
                "collection_id": [self.output["collection_id"]] * len(self.cycles),
                "repeat": [repeat for repeat, _ in self.cycles],
                "phase": [phase for _, phase in self.cycles],
            },
            schema={"collection_id": pl.String, "repeat": pl.Int64, "phase": pl.String},
        )
        runs = metrics.with_columns(pl.lit(self.return_code).alias("return_code"))
        return [FakeMetricsTable(metrics), FakeRunsTable(runs)]


class CollectResultsTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(self.enterContext(TemporaryDirectory()))
        self.events = []
        self.hooks = []
        self.threads = []
        self.fail_start = None
        self.start_error = RuntimeError("worker start failed")
        self.benchmark = FakeBenchmark(self.events, self.assert_ready_to_finalize)
        self.generic = collect.data_collection.GenericCollectorConfig(
            output_dir=str(self.output_dir),
            hooks=[],
            poll_rate=0.001,
            output_interval="1m",
        )
        self.config = SimpleNamespace(generic=self.generic)
        self.enterContext(
            patch.object(type(self.generic), "get_hooks", return_value=self.hooks)
        )
        machine = Mock()
        machine.to_polars.return_value = pl.DataFrame(
            {"machine": [{"uptime_sec": 1, "cores": 2}]}
        )
        self.enterContext(
            patch.object(collect.data_collection, "machine_info", return_value=machine)
        )
        self.enterContext(
            patch.object(collect, "get_user_group_ids", return_value=(1234, 5678))
        )
        self.chown = self.enterContext(patch.object(collect.os, "chown"))
        self.enterContext(patch.object(collect.uuid, "uuid4", return_value="test-id"))
        self.enterContext(patch.object(collect.sys, "stdin", StringIO("")))
        self.enterContext(patch.object(collect, "Thread", side_effect=self.make_thread))
        self.clock_calls = 0
        self.enterContext(patch.object(collect, "datetime", Mock(now=self.now)))
        self.signal_handlers = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGINT, signal.SIGALRM, signal.SIGUSR1)
        }
        self.addCleanup(self.assert_signals_restored)

    def now(self):
        self.events.append(
            "measurement_start" if self.clock_calls == 0 else "measurement_stop"
        )
        self.clock_calls += 1
        return datetime(2026, 1, 1) + timedelta(seconds=7 * (self.clock_calls - 1))

    def make_thread(self, *, target, args):
        role = (
            "poll"
            if target is collect.poll_instrumentation
            else "output"
            if target is collect.output_data_thread
            else "stdin"
        )
        events = self.events

        class TrackedThread(Thread):
            def join(self, timeout=None):
                super().join(timeout)
                if not self.is_alive():
                    events.append(f"{role}:joined")

        def worker():
            events.append(f"{role}:start")
            try:
                target(*args)
            finally:
                events.append(f"{role}:stop")

        thread = TrackedThread(target=worker)
        if role == self.fail_start:
            thread.start = Mock(side_effect=self.start_error)
        self.threads.append((role, thread))
        return thread

    def assert_signals_restored(self):
        for sig, handler in self.signal_handlers.items():
            self.assertEqual(signal.getsignal(sig), handler)

    def assert_ready_to_finalize(self):
        if "measurement_start" in self.events:
            self.assertIn("measurement_stop", self.events)
            self.assertEqual(self.clock_calls, 2)
        self.assertTrue(self.benchmark.clients_stopped)
        for role, thread in self.threads:
            if thread.ident is not None and role != "stdin":
                self.assertFalse(thread.is_alive())
                self.assertIn(f"{role}:joined", self.events)
        for hook in self.hooks:
            if hook.loaded:
                self.assertTrue(hook.closed)

    def run_collection(self, prefix="trial"):
        return collect.run_collect(
            collector_config=self.config,
            benchmark=self.benchmark,
            verbose=False,
            collection_prefix=prefix,
        )

    def assert_saved(self, return_code, error=None, rows=4):
        output = self.benchmark.output
        collection_dir = output["collection_dir"]
        category = "curated" if self.hooks else "baseline"
        self.assertEqual(
            collection_dir,
            self.output_dir / category / "redis" / output["collection_id"],
        )
        system = pl.read_parquet(collection_dir / "system_info.end.parquet")
        self.assertEqual(system["collection_id"].to_list(), [output["collection_id"]])
        self.assertEqual(system["return_code"].to_list(), [return_code])
        self.assertEqual(system["error"].to_list(), [error])
        self.assertEqual(
            system["collection_time_sec"].to_list(), [7.0 if self.clock_calls else 0.0]
        )
        for table in (FakeMetricsTable, FakeRunsTable):
            frame = pl.read_parquet(collection_dir / f"{table.name()}.end.parquet")
            self.assertEqual(frame.height, rows)
            self.assertEqual(
                frame["collection_id"].to_list(), [output["collection_id"]] * rows
            )
        self.assertEqual(self.benchmark.finalize_count, 1)
        self.assert_ready_to_finalize()
        return system

    def test_all_cycles_finalize_once_into_normal_redis_output(self):
        self.hooks.append(FakeHook(self.events))
        flushed = Event()
        original_output = collect.output_collections_to_file

        def output(*args):
            if args[3] != "end":
                self.assertEqual(self.benchmark.finalize_count, 0)
                result = original_output(*args)
                flushed.set()
                return result
            self.events.append("end_write")
            return original_output(*args)

        def after_cycle():
            flushed.clear()
            self.assertTrue(flushed.wait(2), "Periodic probe output did not run")

        self.benchmark.after_cycle = after_cycle
        with (
            patch.object(collect, "timeparse", return_value=0.001),
            patch.object(collect, "output_collections_to_file", side_effect=output),
        ):
            self.assertEqual(self.run_collection(), 0)
        system = self.assert_saved(0)
        self.assertEqual(system["hooks"].to_list(), [["fake_hook"]])
        self.assertEqual(self.benchmark.output["collection_id"], "trial-test-id")
        self.assertEqual(self.benchmark.output["run_label"], "trial")
        self.assertEqual(self.benchmark.output["owner_ids"], (1234, 5678))
        self.chown.assert_any_call(self.benchmark.output["collection_dir"], 1234, 5678)
        self.assertLess(
            self.events.index("setup"), self.events.index("measurement_start")
        )

        self.assertLess(self.events.index("configure"), self.events.index("poll:start"))
        self.assertLess(
            self.events.index("configure"), self.events.index("output:start")
        )
        self.assertLess(self.events.index("configure"), self.events.index("run"))
        self.assertLess(self.events.index("cycle:1:run"), self.events.index("finalize"))
        self.assertLess(
            self.events.index("output:joined"), self.events.index("fake_hook:close")
        )
        self.assertLess(self.events.index("finalize"), self.events.index("end_write"))
        self.assertEqual(self.benchmark.kill_count, 0)

    def test_probe_final_data_failure_does_not_lose_benchmark_tables(self):
        hook = FakeHook(self.events)
        hook.pop_error = RuntimeError("probe final data failed")
        self.hooks.append(hook)
        with self.assertLogs(collect.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "probe final data failed"):
                self.run_collection()
        self.assert_saved(1, "probe final data failed")

    def test_baseline_nonzero_exit_is_saved_and_returned(self):
        self.benchmark.return_code = 9
        self.assertEqual(self.run_collection(prefix=None), 9)
        self.assert_saved(9)
        self.assertEqual(self.benchmark.output["collection_id"], "test-id")
        self.assertIsNone(self.benchmark.output["run_label"])

    def test_cancellation_kills_once_and_saves_partial_results(self):
        def cancel():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            self.assertTrue(self.benchmark.cancelled.wait(2))

        self.benchmark.after_cycle = cancel
        self.assertEqual(self.run_collection(), 1)
        self.assertEqual(self.benchmark.kill_count, 1)
        self.assert_saved(1, rows=1)

    def test_run_failure_saves_partial_results_and_preserves_original_error(self):
        original = RuntimeError("run failed")
        self.benchmark.run_error = original
        self.benchmark.kill_error = RuntimeError("kill failed")
        hook = FakeHook(self.events)
        hook.close_error = RuntimeError("close failed")
        self.hooks.append(hook)
        with self.assertLogs(collect.__name__, level="ERROR") as logs:
            with self.assertRaises(RuntimeError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, original)
        self.assertEqual(self.benchmark.kill_count, 1)
        self.assertTrue(any("kill failed" in entry for entry in logs.output))
        self.assertTrue(any("close failed" in entry for entry in logs.output))
        self.assert_saved(1, "run failed", rows=1)

    def test_poll_failure_is_rethrown_after_partial_results(self):
        original = RuntimeError("poll failed")
        self.benchmark.poll_error = original
        with self.assertLogs(collect.__name__, level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, original)
        self.assertEqual(self.benchmark.kill_count, 1)
        self.assert_saved(1, "poll failed", rows=len(self.benchmark.cycles))

    def test_hook_load_failure_closes_only_loaded_hooks_without_joining(self):
        loaded = FakeHook(self.events)
        failed = FakeHook(self.events, "failed_hook")
        failed.load_error = RuntimeError("load failed")
        self.hooks.extend([loaded, failed])
        with self.assertLogs(collect.__name__, level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, failed.load_error)
        self.assertTrue(loaded.closed)
        self.assertFalse(failed.closed)
        self.assertFalse(self.benchmark.started)
        self.assertEqual(self.threads, [])
        self.assert_saved(1, "load failed", rows=0)

    def test_output_start_failure_cancels_poller_and_saves_results(self):
        self.fail_start = "output"
        self.hooks.append(FakeHook(self.events))
        with self.assertLogs(collect.__name__, level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, self.start_error)
        self.assertFalse(self.benchmark.started)
        self.assertEqual(self.benchmark.kill_count, 1)
        self.assertNotIn("output:joined", self.events)
        self.assert_saved(1, "worker start failed", rows=0)

    def test_poll_start_failure_does_not_join_unstarted_thread(self):
        self.fail_start = "poll"
        self.hooks.append(FakeHook(self.events))
        with self.assertLogs(collect.__name__, level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, self.start_error)
        self.assertNotIn("poll:joined", self.events)
        self.assertEqual(self.benchmark.kill_count, 1)
        self.assert_saved(1, "worker start failed", rows=0)

    def test_periodic_output_failure_cancels_and_is_not_success(self):
        original = OSError("periodic output failed")
        attempted = Event()
        original_output = collect.output_collections_to_file

        def output(*args):
            if args[3] != "end":
                attempted.set()
                raise original
            return original_output(*args)

        self.benchmark.after_cycle = lambda: self.assertTrue(attempted.wait(2))
        with (
            patch.object(collect, "timeparse", return_value=0.001),
            patch.object(collect, "output_collections_to_file", side_effect=output),
            self.assertLogs(collect.__name__, level="ERROR"),
            self.assertRaises(OSError) as raised,
        ):
            self.run_collection()
        self.assertIs(raised.exception, original)
        self.assert_saved(1, "periodic output failed", rows=len(self.benchmark.cycles))

    def test_finalization_failure_still_saves_system_info(self):
        original = ValueError("parse failed")
        self.benchmark.finalize_error = original
        with self.assertLogs(collect.__name__, level="ERROR"):
            with self.assertRaises(ValueError) as raised:
                self.run_collection()
        self.assertIs(raised.exception, original)
        self.assertEqual(self.benchmark.finalize_count, 1)
        system = pl.read_parquet(
            self.benchmark.output["collection_dir"] / "system_info.end.parquet"
        )
        self.assertEqual(system["return_code"].to_list(), [1])
        self.assertEqual(system["error"].to_list(), ["parse failed"])
        self.assert_ready_to_finalize()


class CollectWorkerTests(unittest.TestCase):
    def test_eof_exits_without_spinning_or_cancelling_collection(self):
        run_event = Event()
        run_event.set()
        read = Mock()
        read.readline.side_effect = ["", AssertionError("Read again after EOF")]
        collect.wait_for_END(run_event, read)
        read.readline.assert_called_once_with()
        self.assertTrue(run_event.is_set())

    def test_end_cancels_collection(self):
        run_event = Event()
        run_event.set()
        collect.wait_for_END(run_event, StringIO("continue\nEND\n"))
        self.assertFalse(run_event.is_set())

    def test_output_stop_wakes_long_interval_worker(self):
        run_event = Event()
        run_event.set()
        stop_event = Event()
        with TemporaryDirectory() as directory:
            thread = Thread(
                target=collect.output_data_thread,
                args=(
                    "id",
                    [],
                    "redis",
                    run_event,
                    False,
                    Path(directory),
                    stop_event,
                    60,
                    1234,
                    5678,
                    [],
                ),
                daemon=True,
            )
            thread.start()
            stop_event.set()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())

    def test_poll_and_kill_errors_always_publish_terminal_status(self):
        for source in ("benchmark", "hook", "drain", "kill"):
            with self.subTest(source=source):
                error = RuntimeError(f"{source} failed")
                benchmark = Mock()
                benchmark.name.return_value = "redis"
                benchmark.poll.return_value = 0
                hook = Mock()
                run_event = Event()
                run_event.set()
                if source == "benchmark":
                    benchmark.poll.side_effect = error
                elif source == "hook":
                    hook.poll.side_effect = error
                elif source == "drain":
                    hook.poll.side_effect = [None, error]
                else:
                    run_event.clear()
                    benchmark.kill.side_effect = error
                queue = Queue(maxsize=1)
                with self.assertLogs(collect.__name__, level="ERROR"):
                    thread = Thread(
                        target=collect.poll_instrumentation,
                        args=(benchmark, [hook], queue, run_event, 0),
                        daemon=True,
                    )
                    thread.start()
                    status = queue.get(timeout=2)
                    thread.join(timeout=1)
                self.assertIs(status, error)
                self.assertFalse(thread.is_alive())

    def test_interrupted_faux_is_zero_other_benchmarks_are_one(self):
        for name, expected in (("faux", 0), ("redis", 1)):
            with self.subTest(benchmark=name):
                benchmark = Mock()
                benchmark.name.return_value = name
                queue = Queue(maxsize=1)
                result = collect.poll_instrumentation(benchmark, [], queue, Event(), 0)
                self.assertEqual(result, expected)
                self.assertEqual(queue.get_nowait(), expected)
                benchmark.kill.assert_called_once_with()

    def test_other_benchmark_output_defaults_are_noops(self):
        benchmark = FauxBenchmark(
            generic_config=GenericBenchmarkConfig(), config=FauxBenchmarkConfig()
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            benchmark.configure_output(
                collection_id="id",
                collection_dir=output,
                run_label="label",
                owner_ids=(1234, 5678),
            )
            self.assertFalse(output.exists())
            first = benchmark.finalize()
            self.assertEqual(first, [])
            self.assertEqual(benchmark.finalize(), [])
            self.assertIsNot(first, benchmark.finalize())

    def test_cli_propagates_collection_exit_code(self):
        config = SimpleNamespace(collector_config=Mock(), benchmark_config=Mock())
        config_factory = Mock()
        config_factory.return_value.merge.return_value = config
        config_file = Mock()
        config_file.read_text.return_value = "{}"
        with (
            patch.object(cli_module, "KernmlopsConfig", config_factory),
            patch.dict(cli_module.benchmarks, {"redis": Mock()}),
            patch.object(collect, "run_collect", return_value=7),
            self.assertRaises(SystemExit) as raised,
        ):
            cli_module.cli_collect_data.callback(
                config_file=config_file,
                benchmark_name="redis",
                verbose=False,
                collection_prefix=None,
            )
        self.assertEqual(raised.exception.code, 7)


if __name__ == "__main__":
    unittest.main()
