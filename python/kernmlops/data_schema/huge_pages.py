import polars as pl
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)


class TraceMMKhugepagedScanPMDDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "trace_mm_khugepaged_scan_pmd"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TraceMMKhugepagedScanPMDDataTable":
        return TraceMMKhugepagedScanPMDDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
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

class CollapseHugePageDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "collapse_huge_page"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CollapseHugePageDataTable":
        return CollapseHugePageDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
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

class TraceMMCollapseHugePageDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "trace_mm_collapse_huge_page"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TraceMMCollapseHugePageDataTable":
        return TraceMMCollapseHugePageDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
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
