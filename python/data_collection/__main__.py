"""CLI"""

from functools import cache
from time import sleep

import bpf_instrumentation
import polars as pl
from system_info import machine_info


@cache
def bpf_hooks() -> list[bpf_instrumentation.BPFProgram]:
    return [
        bpf_instrumentation.QuantaRuntimeBPFHook(),
    ]

def poll_instrumentation(bpf_programs: list[bpf_instrumentation.BPFProgram]):
    while (1):
        try:
            for bpf_program in bpf_programs:
                bpf_program.poll()
            sleep(.5)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    system_info = machine_info().to_polars()
    collection_id = system_info["identification"][0]["collection_id"]
    print(system_info.unnest(system_info.columns))

    bpf_programs = bpf_hooks()
    for bpf_program in bpf_programs:
        bpf_program.load()
    print("Finished loading BPF programs")

    poll_instrumentation(bpf_programs)

    bpf_dfs = [
        bpf_program.pop_data().with_columns(pl.lit(collection_id).alias("collection_id"))
        for bpf_program in bpf_programs
    ]
    for bpf_df in bpf_dfs:
        print(bpf_df)
