import polars as pl
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)
from typing_extensions import (
    _ProtocolMeta,  # pyright: ignore [reportAttributeAccessIssue]
)


class GenericTableMeta(_ProtocolMeta):
    def __new__(cls, name, bases, dct):
        probe_name = dct.get('probe_name')

        def name_func(cls) -> str:
            return f"{probe_name}"

        def schema(cls) -> pl.Schema:
            return pl.Schema()

        def from_df(cls, table: pl.DataFrame) -> name: # pyright: ignore [reportInvalidTypeForm]
            return cls(table=table)

        def __init__(self, table: pl.DataFrame):
            self._table = table

        def table(self) -> pl.DataFrame:
            return self._table

        def filtered_table(self) -> pl.DataFrame:
            return self.table

        def graphs(self) -> list[type[CollectionGraph]]:
            return []

        def by_pid(self, pids: int | list[int]) -> pl.DataFrame:
            if isinstance(pids, int):
                pids = [pids]
            return self.filtered_table().filter(pl.col("pid").is_in(pids))

        dct["__init__"] = __init__
        dct["name"] = classmethod(name_func)
        dct["from_df"] = classmethod(from_df)
        dct["table"] = property(table)
        dct["filtered_table"] = filtered_table
        dct["graphs"] = graphs
        dct["by_pid"] = by_pid
        return super().__new__(cls, name, bases, dct)

class ProcessMetadataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "process_metadata"

class ProcessTraceDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "process_trace"

class TraceMMRSSStatDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "mm_rss_stat"

class ZswapRuntimeDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "zswap_runtime"

class TraceMMKhugepagedScanPMDDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "trace_mm_khugepaged_scan_pmd"

class CollapseHugePageDataTableRaw(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "collapse_huge_pages"

class TraceMMCollapseHugePageDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "trace_mm_collapse_huge_page"

class CBMMEagerDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "cbmm_eager"

class CBMMPrezeroingDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "cbmm_prezero"

class MadviseDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "madvise"

class UnmapRangeDataTable(CollectionTable, metaclass=GenericTableMeta):
    probe_name = "unmap_range"
