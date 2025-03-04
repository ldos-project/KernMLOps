import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import sleep
from typing import cast

import data_collection
import data_schema
import polars as pl
from data_schema import demote
from kernmlops_benchmark import (
    Benchmark,
    BenchmarkNotConfiguredError,
    BenchmarkNotRunningError,
)
from kernmlops_config import ConfigBase


def wait_for_END(run_event: Event, read):
    while run_event.is_set() and "END" not in read.readline():
        continue
    run_event.clear()

def poll_instrumentation(
  benchmark: Benchmark,
  bpf_programs: list[data_collection.bpf.BPFProgram],
  queue: Queue,
  run_event: Event,
  poll_rate: float = .5,
) -> int:

    return_code = None
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

    if not run_event.is_set():
        benchmark.kill()
        return_code = 0 if benchmark.name() == "faux" else 1

    # Poll again to clean out all buffers
    for bpf_program in bpf_programs:
        try:
            bpf_program.poll()
        except Exception:
            pass
    return_code = return_code if return_code is not None else 1
    queue.put(return_code)
    return return_code

def signal_handler_factory(event: Event):
    return lambda x,y: event.clear()

def run_collect(
    *,
    collector_config: ConfigBase,
    benchmark: Benchmark,
    verbose: bool
):
    if not benchmark.is_configured():
        raise BenchmarkNotConfiguredError(f"benchmark {benchmark.name()} is not configured")
    benchmark.setup()

    generic_config = cast(data_collection.GenericCollectorConfig, getattr(collector_config, "generic"))
    bpf_programs = generic_config.get_hooks()
    system_info = data_collection.machine_info().to_polars()
    system_info = system_info.unnest(system_info.columns)
    collection_id = system_info["collection_id"][0]
    output_dir = generic_config.get_output_dir() / "curated" if bpf_programs else generic_config.get_output_dir() / "baseline"
    queue = Queue(maxsize=1)
    run_event = Event()
    run_event.set()

    for bpf_program in bpf_programs:
        bpf_program.load(collection_id)
        if verbose:
            print(f"{bpf_program.name()} BPF program loaded")
    if verbose:
        print("Finished loading BPF programs")

    # Configure signal capture
    signal.signal(signal.SIGINT, signal_handler_factory(run_event))
    signal.signal(signal.SIGALRM, signal_handler_factory(run_event))
    signal.signal(signal.SIGUSR1, signal_handler_factory(run_event))

    # Create stdin killer daemon
    read_thread = Thread(target = wait_for_END, args = (run_event, sys.stdin))
    read_thread.daemon = True
    read_thread.start()

    # Create polling thread
    poll_thread = Thread(target = poll_instrumentation, args = (benchmark, bpf_programs, queue, run_event, generic_config.poll_rate))
    poll_thread.start()

    tick = datetime.now()

    benchmark.run()

    if verbose:
        print(f"Started benchmark {benchmark.name()}")
    return_code = queue.get()

    collection_time_sec = (datetime.now() - tick).total_seconds()
    poll_thread.join()
    for bpf_program in bpf_programs:
        bpf_program.close()

    demote()()
    if verbose:
        print(f"Benchmark ran for {collection_time_sec}s")
    if return_code != 0:
        print(f"Benchmark {benchmark.name()} failed with return code {return_code}")
        output_dir = generic_config.get_output_dir() / "failed"


    collection_tables: list[data_schema.CollectionTable] = [
        data_schema.SystemInfoTable.from_df(
            system_info.with_columns([
                pl.lit(collection_time_sec).alias("collection_time_sec"),
                pl.lit(os.getpid()).alias("collection_pid"),
                pl.lit(benchmark.name()).alias("benchmark_name"),
                pl.lit([hook.name() for hook in bpf_programs]).cast(pl.List(pl.String())).alias("hooks"),
            ])
        )
    ]
    for bpf_program in bpf_programs:
        collection_tables.extend(bpf_program.pop_data())
    for collection_table in collection_tables:
        with pl.Config(tbl_cols=-1):
            if verbose:
                print(f"{collection_table.name()}: {collection_table.table}")
        Path(output_dir / collection_table.name()).mkdir(parents=True, exist_ok=True)
        collection_table.table.write_parquet(output_dir / collection_table.name() / f"{collection_id}.{benchmark.name()}.parquet")
    collection_data = data_schema.CollectionData.from_tables(collection_tables)
    if generic_config.output_graphs:
        collection_data.graph(out_dir=generic_config.get_output_dir() / "graphs")
