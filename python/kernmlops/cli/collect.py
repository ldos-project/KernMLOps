from pathlib import Path
from time import sleep

import data_collection
import polars as pl


def poll_instrumentation(
  bpf_programs: list[data_collection.bpf.BPFProgram],
  poll_rate: float = .5,
):
    while (1):
        try:
            for bpf_program in bpf_programs:
                bpf_program.poll()
            sleep(poll_rate)
            # clean data when missed samples - or detect?
            # include collector pid
        except KeyboardInterrupt:
            break


def run_collect(
    output_dir: Path,
    bpf_programs: list[data_collection.bpf.BPFProgram],
    poll_rate: float,
    verbose: bool
):
    benchmark = "linux_build"
    system_info = data_collection.machine_info().to_polars()
    system_info = system_info.unnest(system_info.columns)
    collection_id = system_info["collection_id"][0]

    for bpf_program in bpf_programs:
        bpf_program.load()
        if verbose:
            print(f"{bpf_program.name()} BPF program loaded")
    if verbose:
        print("Finished loading BPF programs")

    poll_instrumentation(bpf_programs, poll_rate=poll_rate)

    bpf_dfs = {
        bpf_program.name(): bpf_program.pop_data().with_columns(pl.lit(collection_id).alias("collection_id"))
        for bpf_program in bpf_programs
    }
    bpf_dfs["system_info"] = system_info
    for bpf_name, bpf_df in bpf_dfs.items():
        if verbose:
            print(f"{bpf_name}: {bpf_df}")
        Path(output_dir / bpf_name).mkdir(parents=True, exist_ok=True)
        bpf_df.write_parquet(output_dir / bpf_name / f"{collection_id}.{benchmark}.parquet")
