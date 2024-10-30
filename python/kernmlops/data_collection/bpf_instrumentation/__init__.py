"""Module for maintaining different BPF hooks/instrumentation."""

from typing import Final, Mapping

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_collection.bpf_instrumentation.file_data_hook import FileDataBPFHook
from data_collection.bpf_instrumentation.memory_usage_hook import MemoryUsageHook
from data_collection.bpf_instrumentation.process_metadata_hook import (
    ProcessMetadataHook,
)
from data_collection.bpf_instrumentation.quanta_runtime_hook import QuantaRuntimeBPFHook
from data_collection.bpf_instrumentation.tlb_perf_hook import (
    CustomHWConfigManager,
    TLBPerfBPFHook,
)

all_hooks: Final[Mapping[str, type[BPFProgram]]] = {
    FileDataBPFHook.name(): FileDataBPFHook,
    MemoryUsageHook.name(): MemoryUsageHook,
    ProcessMetadataHook.name(): ProcessMetadataHook,
    QuantaRuntimeBPFHook.name(): QuantaRuntimeBPFHook,
    TLBPerfBPFHook.name(): TLBPerfBPFHook,
}


def hook_names() -> list[str]:
    return list(all_hooks.keys())


__all__ = [
    "all_hooks",
    "hook_names",
    "BPFProgram",
    "CustomHWConfigManager",
    "QuantaRuntimeBPFHook",
]
