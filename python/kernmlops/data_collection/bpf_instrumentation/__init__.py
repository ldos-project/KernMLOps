"""Module for maintaining different BPF hooks/instrumentation."""

from functools import cache

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_collection.bpf_instrumentation.quanta_runtime_hook import QuantaRuntimeBPFHook

all_hooks = [
    QuantaRuntimeBPFHook,
]


@cache
def hooks() -> list[BPFProgram]:
    return [
        QuantaRuntimeBPFHook(),
    ]


__all__ = [
    "all_hooks",
    "hooks",
    "BPFProgram",
    "QuantaRuntimeBPFHook",
]
