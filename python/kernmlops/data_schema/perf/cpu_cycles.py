import polars as pl
from bcc import (PerfType, PerfHWConfig)
from data_schema.perf.perf_schema import (
    CustomHWEventID,
    PerfCollectionTable,
    PerfHWCacheConfig,
    #PerfHWConfig,
)
from data_schema.schema import CollectionGraph

class CPUCylesPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "cpu_cycles" # cosmetic
    # From perf_hw_id in uapi/linux/perf_event.h
    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HARDWARE

    @classmethod
    def ev_config(cls) -> int:
        return PerfHWConfig.CPU_CYCLES

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

    @classmethod
    def component_name(cls) -> str:
        return "cpu_cycles" # cosmetic

    @classmethod
    def measured_event_name(cls) -> str:
        return "cpu_cycles" # cosmetic

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CPUCyclesPerfTable":
        return CPUCylesPerfTable(
            table=table.cast(cls.schema(), strict=True)
        )  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []