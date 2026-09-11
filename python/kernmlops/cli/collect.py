import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import sleep
from typing import cast

import data_collection
import data_schema
import polars as pl
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import get_user_group_ids
from kernmlops_benchmark import (
    Benchmark,
    BenchmarkNotConfiguredError,
    BenchmarkNotRunningError,
)
from kernmlops_config import ConfigBase
from pytimeparse.timeparse import timeparse

logger = logging.getLogger(__name__)


def wait_for_END(run_event: Event, read):
    while run_event.is_set():
        line = read.readline()
        if not line:
            return
        if "END" in line:
            run_event.clear()
            return


def poll_instrumentation(
    benchmark: Benchmark,
    bpf_programs: list[data_collection.bpf.BPFProgram],
    queue: Queue,
    run_event: Event,
    poll_rate: float = 0.5,
) -> int:
    return_code = None
    error: BaseException | None = None
    try:
        while return_code is None and run_event.is_set():
            try:
                for bpf_program in bpf_programs:
                    bpf_program.poll()
                if poll_rate > 0:
                    sleep(poll_rate)
                return_code = benchmark.poll()
                # clean data when missed samples - or detect?
            except BenchmarkNotRunningError:
                continue
    except BaseException as exc:
        error = exc
        run_event.clear()
        logger.exception("Benchmark polling failed")
    finally:
        try:
            if not run_event.is_set():
                return_code = 0 if benchmark.name() == "faux" else 1
                try:
                    benchmark.kill()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    logger.exception("Benchmark cancellation failed")

            # Poll again to clean out all buffers.
            for bpf_program in bpf_programs:
                try:
                    bpf_program.poll()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    logger.exception("Final instrumentation poll failed")
        finally:
            return_code = return_code if return_code is not None else 1
            queue.put(error if error is not None else return_code)
    return return_code


def signal_handler_factory(event: Event):
    return lambda x, y: event.clear()


def output_collections_to_file(
    collection_id: str,
    collection_tables: list[data_schema.CollectionTable],
    bpf_programs: list[BPFProgram],
    name: str,
    benchmark_name: str,
    verbose: bool,
    output_dir: Path,
    ids: tuple[int, int] | None = None,
):
    for bpf_program in bpf_programs:
        collection_tables.extend(bpf_program.pop_data())
    for collection_table in collection_tables:
        with pl.Config(tbl_cols=-1):
            if verbose:
                print(f"{collection_table.name()}: {collection_table.table}")
        full_path = Path(
            output_dir
            / benchmark_name
            / collection_id
            / f"{collection_table.name()}.{name}.parquet"
        )
        collection_table.table.write_parquet(full_path)
        if ids is not None:
            os.chown(full_path, ids[0], ids[1])
    return collection_tables


def output_data_thread(
    collection_id: str,
    bpf_programs: list[BPFProgram],
    benchmark_name: str,
    run_event: Event,
    verbose: bool,
    output_dir: Path,
    stop_event: Event,
    output_interval: int | float,
    user_id: int,
    group_id: int,
    errors: list[BaseException],
):
    num: int = 0
    while not stop_event.wait(output_interval):
        if not run_event.is_set():
            return
        try:
            output_collections_to_file(
                collection_id,
                [],
                bpf_programs,
                str(num),
                benchmark_name,
                verbose,
                output_dir,
                (user_id, group_id),
            )
        except BaseException as exc:
            errors.append(exc)
            run_event.clear()
            logger.exception("Periodic collection output failed")
            return
        num += 1


def run_collect(
    *,
    collector_config: ConfigBase,
    benchmark: Benchmark,
    verbose: bool,
    collection_prefix: str | None,
):
    if not benchmark.is_configured():
        raise BenchmarkNotConfiguredError(
            f"benchmark {benchmark.name()} is not configured"
        )
    benchmark.setup()

    generic_config = cast(
        data_collection.GenericCollectorConfig, getattr(collector_config, "generic")
    )
    bpf_programs = generic_config.get_hooks()
    system_info = data_collection.machine_info().to_polars()
    system_info = system_info.unnest(system_info.columns)
    collection_id = str(uuid.uuid4())
    if collection_prefix:
        collection_id = f"{collection_prefix}-{collection_id}"
    output_dir = (
        generic_config.get_output_dir() / "curated"
        if bpf_programs
        else generic_config.get_output_dir() / "baseline"
    )
    collection_dir = output_dir / benchmark.name() / collection_id
    (user_id, group_id) = get_user_group_ids()
    collection_dir.mkdir(parents=True, exist_ok=True)
    os.chown(generic_config.get_output_dir(), user_id, group_id)
    os.chown(output_dir, user_id, group_id)
    os.chown(collection_dir.parent, user_id, group_id)
    os.chown(collection_dir, user_id, group_id)
    benchmark.configure_output(
        collection_id=collection_id,
        collection_dir=collection_dir,
        run_label=collection_prefix,
        owner_ids=(user_id, group_id),
    )

    queue = Queue(maxsize=1)
    run_event = Event()
    run_event.set()
    output_stop = Event()
    output_errors: list[BaseException] = []
    loaded_programs: list[BPFProgram] = []
    poll_thread: Thread | None = None
    output_thread: Thread | None = None
    previous_handlers = {}
    tick: datetime | None = None
    return_code = 1
    error: BaseException | None = None

    def record_error(exc: BaseException) -> None:
        nonlocal error, return_code
        if error is None:
            error = exc
        return_code = return_code or 1
        logger.error("Collection failed", exc_info=(type(exc), exc, exc.__traceback__))

    try:
        for bpf_program in bpf_programs:
            bpf_program.load(collection_id)
            loaded_programs.append(bpf_program)
            if verbose:
                print(f"{bpf_program.name()} BPF program loaded")
        if verbose:
            print("Finished loading BPF programs")

        for sig in (signal.SIGINT, signal.SIGALRM, signal.SIGUSR1):
            previous_handlers[sig] = signal.signal(
                sig, signal_handler_factory(run_event)
            )

        read_thread = Thread(target=wait_for_END, args=(run_event, sys.stdin))
        read_thread.daemon = True
        read_thread.start()

        poll_thread = Thread(
            target=poll_instrumentation,
            args=(
                benchmark,
                loaded_programs,
                queue,
                run_event,
                generic_config.poll_rate,
            ),
        )
        poll_thread.start()

        output_interval_parse = timeparse(generic_config.output_interval)
        output_interval = 60 if output_interval_parse is None else output_interval_parse
        output_thread = Thread(
            target=output_data_thread,
            args=(
                collection_id,
                loaded_programs,
                benchmark.name(),
                run_event,
                generic_config.output_dfs,
                output_dir,
                output_stop,
                output_interval,
                user_id,
                group_id,
                output_errors,
            ),
        )
        output_thread.daemon = True
        output_thread.start()

        tick = datetime.now()
        if run_event.is_set():
            benchmark.run()
            if verbose:
                print(f"Started benchmark {benchmark.name()}")
        status = queue.get()
        if isinstance(status, BaseException):
            raise status
        return_code = status
    except BaseException as exc:
        record_error(exc)
    finally:
        run_event.clear()
        # Once started, the poller alone owns cancellation, including run() failures.
        if poll_thread is not None:
            if poll_thread.ident is not None:
                poll_thread.join()
            else:
                try:
                    benchmark.kill()
                except BaseException as exc:
                    record_error(exc)
        collection_time_sec = (
            (datetime.now() - tick).total_seconds() if tick is not None else 0.0
        )
        output_stop.set()
        if output_thread is not None and output_thread.ident is not None:
            output_thread.join()
        for exc in output_errors:
            record_error(exc)
        for bpf_program in loaded_programs:
            try:
                bpf_program.close()
            except BaseException as exc:
                record_error(exc)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if verbose:
        print(f"Benchmark ran for {collection_time_sec}s")

    collection_tables: list[data_schema.CollectionTable] = []
    try:
        collection_tables.extend(benchmark.finalize())
    except BaseException as exc:
        record_error(exc)
    # A failed probe must not discard finalized workload results or the other
    # probes' tables. Record these failures before constructing system metadata.
    for bpf_program in loaded_programs:
        try:
            collection_tables.extend(bpf_program.pop_data())
        except BaseException as exc:
            record_error(exc)
    try:
        collection_tables.insert(
            0,
            data_schema.SystemInfoTable.from_df_id(
                system_info.with_columns(
                    [
                        pl.lit(collection_time_sec).alias("collection_time_sec"),
                        pl.lit(os.getpid()).alias("collection_pid"),
                        pl.lit(benchmark.name()).alias("benchmark_name"),
                        pl.lit(return_code).alias("return_code"),
                        pl.lit(
                            str(error) if error is not None else None, dtype=pl.String
                        ).alias("error"),
                        pl.lit([hook.name() for hook in loaded_programs])
                        .cast(pl.List(pl.String()))
                        .alias("hooks"),
                    ]
                ),
                collection_id,
            ),
        )
        collection_tables = output_collections_to_file(
            collection_id,
            collection_tables,
            [],
            "end",
            benchmark.name(),
            generic_config.output_dfs,
            output_dir,
            (user_id, group_id),
        )
    except BaseException as exc:
        record_error(exc)
    print(f"{collection_id}")
    if error is not None:
        raise error
    collection_data = data_schema.CollectionData.from_tables(collection_tables)

    if generic_config.output_graphs:
        collection_data.graph(out_dir=generic_config.get_output_dir() / "graphs")
    return return_code
