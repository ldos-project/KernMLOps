from typing import Mapping

from data_schema.perf.hw_cache.branch_perf import BranchPerfTable

# from data_schema.perf.hw_cache.tlb_perf import DTLBPerfTable, ITLBPerfTable, TLBFlushPerfTable
from data_schema.perf.hw_cache.l1d_perf import L1DPerfTable

# from data_schema.perf.hw_cache.l1i_perf import L1IPerfTable
from data_schema.perf.hw_cache.llc_perf import LLCPerfTable
from data_schema.perf.hw_cache.local_perf import LocalMemPerfTable
from data_schema.perf.perf_schema import (
    CustomHWEventID,
    PerfCollectionTable,
)

# from data_schema.perf.hardware.cycles_perf import CPUCyclesPerfTable, InstructionsPerfTable

perf_table_types: Mapping[str, type[PerfCollectionTable]] = {
    # DTLBPerfTable.name(): DTLBPerfTable,
    # ITLBPerfTable.name(): ITLBPerfTable,
    # TLBFlushPerfTable.name(): TLBFlushPerfTable,
    L1DPerfTable.name(): L1DPerfTable,
    # L1IPerfTable.name(): L1IPerfTable,
    LLCPerfTable.name(): LLCPerfTable,
    LocalMemPerfTable.name(): LocalMemPerfTable,
    BranchPerfTable.name(): BranchPerfTable,
    # CPUCyclesPerfTable.name(): CPUCyclesPerfTable,
    # InstructionsPerfTable.name(): InstructionsPerfTable,
}

__all__ = [
  "perf_table_types",
  "CustomHWEventID",
  "PerfCollectionTable",
]
