from typing import Mapping

from data_schema.perf.perf_schema import (
    CustomHWEventID,
    PerfCollectionTable,
)
from data_schema.perf.tlb_perf import (
    DTLBPerfTable,
    DTLBWalkDurationPerfTable,
    ITLBPerfTable,
    TLBFlushPerfTable,
)

perf_table_types: Mapping[str, type[PerfCollectionTable]] = {
    DTLBPerfTable.name(): DTLBPerfTable,
    ITLBPerfTable.name(): ITLBPerfTable,
    TLBFlushPerfTable.name(): TLBFlushPerfTable,
    DTLBWalkDurationPerfTable.name(): DTLBWalkDurationPerfTable,
}

__all__ = [
  "perf_table_types",
  "CustomHWEventID",
  "PerfCollectionTable",
]
