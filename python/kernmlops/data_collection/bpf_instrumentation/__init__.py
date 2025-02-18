"""Module for maintaining different BPF hooks/instrumentation."""

from typing import Final, Mapping

from data_collection.bpf_instrumentation.blk_io_hook import BlockIOBPFHook
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_collection.bpf_instrumentation.cbmm import (
    CBMMBPFHook,
)
from data_collection.bpf_instrumentation.collapse_huge_page import (
    CollapseHugePageBPFHook,
)
from data_collection.bpf_instrumentation.file_data_hook import FileDataBPFHook
from data_collection.bpf_instrumentation.fork_and_exit import TraceProcessHook
from data_collection.bpf_instrumentation.memory_usage_hook import MemoryUsageHook
from data_collection.bpf_instrumentation.mm_rss_stat import TraceRSSStatBPFHook
from data_collection.bpf_instrumentation.perf import (
    CustomHWConfigManager,
    PerfBPFHook,
)
from data_collection.bpf_instrumentation.process_metadata_hook import (
    ProcessMetadataHook,
)
from data_collection.bpf_instrumentation.quanta_runtime_hook import QuantaRuntimeBPFHook

all_hooks: Final[Mapping[str, type[BPFProgram]]] = {
    FileDataBPFHook.name(): FileDataBPFHook,
    MemoryUsageHook.name(): MemoryUsageHook,
    ProcessMetadataHook.name(): ProcessMetadataHook,
    QuantaRuntimeBPFHook.name(): QuantaRuntimeBPFHook,
    BlockIOBPFHook.name(): BlockIOBPFHook,
    PerfBPFHook.name(): PerfBPFHook,
    CollapseHugePageBPFHook.name(): CollapseHugePageBPFHook,
    CBMMBPFHook.name(): CBMMBPFHook,
    TraceRSSStatBPFHook.name(): TraceRSSStatBPFHook,
    TraceProcessHook.name(): TraceProcessHook,
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
