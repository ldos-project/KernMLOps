import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from time import sleep
from typing import cast

import data_collection
import data_schema
import polars as pl
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import demote, get_user_group_ids
from kernmlops_benchmark import (
    Benchmark,
    BenchmarkNotConfiguredError,
    BenchmarkNotRunningError,
)
from kernmlops_config import ConfigBase
from pytimeparse.timeparse import timeparse


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

def output_collections_to_file(collection_id: str, collection_tables : list[data_schema.CollectionTable], bpf_programs: list[BPFProgram], name: str,
                               benchmark_name: str, verbose: bool, output_dir: Path, ids: tuple[int,int] | None = None):
    for bpf_program in bpf_programs:
        collection_tables.extend(bpf_program.pop_data())
    for collection_table in collection_tables:
        with pl.Config(tbl_cols=-1):
            if verbose:
                print(f"{collection_table.name()}: {collection_table.table}")
        full_path = Path(output_dir/benchmark_name/collection_id/f"{collection_table.name()}.{name}.parquet")
        collection_table.table.write_parquet(full_path)
        if ids is not None:
            os.chown(full_path, ids[0], ids[1])
    return collection_tables

def output_data_thread(collection_id: str, bpf_programs: list[BPFProgram], benchmark_name: str, run_event: Event,
                       verbose: bool, output_dir: Path, lock: Lock, ended: bool, output_interval: int | float, user_id: int, group_id: int):
    num : int = 0
    sleep(output_interval)
    while run_event.is_set():
        lock.acquire()
        try:
            if(ended):
                lock.release()
                return
            output_collections_to_file(collection_id, [], bpf_programs, str(num), benchmark_name, verbose, output_dir, (user_id, group_id))
        except Exception as e:
            print(e)
        lock.release()
        num += 1
        sleep(output_interval)

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

    # Create output thread
    output_interval_parse : int | float | None = timeparse(generic_config.output_interval)
    output_interval = 60
    if output_interval_parse is not None:
        output_interval = output_interval_parse
    ended = False
    output_lock = Lock()
    (user_id, group_id) = get_user_group_ids()
    Path(output_dir/benchmark.name()/collection_id).mkdir(parents=True, exist_ok=True)
    os.chown(generic_config.get_output_dir(), user_id, group_id)
    os.chown(output_dir, user_id, group_id)
    os.chown(Path(output_dir/benchmark.name()), user_id, group_id)
    os.chown(Path(output_dir/benchmark.name()/collection_id), user_id, group_id)
    output_thread = Thread(target = output_data_thread, args = (collection_id, bpf_programs, benchmark.name(), run_event, verbose, output_dir, output_lock,
                                                                ended, output_interval, user_id, group_id))
    output_thread.daemon = True
    output_thread.start()



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

    output_lock.acquire()
    ended = True
    collection_tables = output_collections_to_file(collection_id, collection_tables, bpf_programs, "end", benchmark.name(), verbose, output_dir)
    output_lock.release()
    collection_data = data_schema.CollectionData.from_tables(collection_tables)

    if generic_config.output_graphs:
        collection_data.graph(out_dir=generic_config.get_output_dir() / "graphs")
    print(f"{collection_id}")
    return return_code
