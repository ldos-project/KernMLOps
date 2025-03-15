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

class CollapseHugePageDataTableRaw(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "collapse_huge_page"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CollapseHugePageDataTableRaw":
        return CollapseHugePageDataTableRaw(table=table)

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


class CollapseHugePageDataTable(CollectionTable):
    """Best effort merged table of CollapseHugePageDataTableRaw and TraceMMCollapseHugePageDataTable."""

    @classmethod
    def name(cls) -> str:
        return "collapse_hugepage"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CollapseHugePageDataTable":
        return CollapseHugePageDataTable(table=table) #.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    @classmethod
    def from_tables(cls, collapse_table: CollapseHugePageDataTableRaw, trace_mm_table: TraceMMCollapseHugePageDataTable) -> "CollapseHugePageDataTable":
        collapse_df = collapse_table.filtered_table().sort("start_ts_ns", descending=False)
        trace_mm_df = trace_mm_table.filtered_table().sort("start_ts_ns", descending=False)
        if trace_mm_df.row(0, named=True)["start_ts_ns"] < collapse_df.row(0, named=True)["start_ts_ns"]:
            collapse_df = collapse_df[1:]
        assert len(collapse_df) == len(trace_mm_df)
        collapse_df = collapse_df.drop([
            "end_ts_ns",
        ])
        trace_mm_df = trace_mm_df.drop([
            "pid",
            "tgid",
            "start_ts_ns",
            "end_ts_ns",
            "mm",
            "collection_id",
        ])
        return cls.from_df(pl.concat([collapse_df, trace_mm_df], how="horizontal"))

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []
