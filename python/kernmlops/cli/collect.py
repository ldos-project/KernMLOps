import os
from datetime import datetime
from pathlib import Path
from time import sleep

import data_collection
import data_schema
import polars as pl
from data_schema import demote
from kernmlops_benchmark import Benchmark, BenchmarkNotConfiguredError


def poll_instrumentation(
  benchmark: Benchmark,
  bpf_programs: list[data_collection.bpf.BPFProgram],
  poll_rate: float = .5,
) -> int:
    return_code = benchmark.poll()
    while return_code is None:
        try:
            for bpf_program in bpf_programs:
                bpf_program.poll()
            sleep(poll_rate)
            return_code = benchmark.poll()
            # clean data when missed samples - or detect?
        except KeyboardInterrupt:
            benchmark.kill()
            return_code = 0 if benchmark.name() == "faux" else 1
    return return_code


def run_collect(
    *,
    data_dir: Path,
    benchmark: Benchmark,
    bpf_programs: list[data_collection.bpf.BPFProgram],
    poll_rate: float,
    verbose: bool
):
    if not benchmark.is_configured():
        raise BenchmarkNotConfiguredError(f"benchmark {benchmark.name()} is not configured")
    benchmark.setup()

    system_info = data_collection.machine_info().to_polars()
    system_info = system_info.unnest(system_info.columns)
    collection_id = system_info["collection_id"][0]
    output_dir = data_dir / "curated" if bpf_programs else data_dir / "baseline"

    for bpf_program in bpf_programs:
        bpf_program.load(collection_id)
        if verbose:
            print(f"{bpf_program.name()} BPF program loaded")
    if verbose:
        print("Finished loading BPF programs")
    benchmark.run()
    if verbose:
        print(f"Started benchmark {benchmark.name()}")

    tick = datetime.now()
    return_code = poll_instrumentation(benchmark, bpf_programs, poll_rate=poll_rate)
    collection_time_sec = (datetime.now() - tick).total_seconds()
    for bpf_program in bpf_programs:
        bpf_program.close()
    demote()()
    if verbose:
        print(f"Benchmark ran for {collection_time_sec}s")
    if return_code != 0:
        print(f"Benchmark {benchmark.name()} failed with return code {return_code}")
        output_dir = data_dir / "failed"


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
        if verbose:
            print(f"{collection_table.name()}: {collection_table.table}")
        Path(output_dir / collection_table.name()).mkdir(parents=True, exist_ok=True)
        collection_table.table.write_parquet(output_dir / collection_table.name() / f"{collection_id}.{benchmark.name()}.parquet")
    collection_data = data_schema.CollectionData.from_tables(collection_tables)
    collection_data.graph(out_dir=data_dir / "graphs")
