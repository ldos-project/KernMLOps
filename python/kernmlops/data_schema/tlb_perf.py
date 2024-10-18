import polars as pl

from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    PerfCollectionTable,
    RatePerfGraph,
)


class DTLBPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "dtlb_misses"

    @classmethod
    def component_name(cls) -> str:
        return "dTLB"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Misses"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "DTLBPerfTable":
        return DTLBPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [DTLBRateGraph]


class DTLBRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return DTLBPerfTable

    @classmethod
    def with_collection(cls, collection_data: CollectionData) -> CollectionGraph | None:
        perf_table = collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return DTLBRateGraph(
                collection_data=collection_data,
                perf_table=perf_table
            )
        return None


class ITLBPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "itlb_misses"

    @classmethod
    def component_name(cls) -> str:
        return "iTLB"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Misses"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "ITLBPerfTable":
        return ITLBPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [ITLBRateGraph]


class ITLBRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return ITLBPerfTable

    @classmethod
    def with_collection(cls, collection_data: CollectionData) -> CollectionGraph | None:
        perf_table = collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return ITLBRateGraph(
                collection_data=collection_data,
                perf_table=perf_table
            )
        return None


class TLBFlushPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "tlb_flushes"

    @classmethod
    def component_name(cls) -> str:
        return "TLB"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Flushes"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TLBFlushPerfTable":
        return TLBFlushPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [TLBFlushRateGraph]


class TLBFlushRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return TLBFlushPerfTable

    @classmethod
    def with_collection(cls, collection_data: CollectionData) -> CollectionGraph | None:
        perf_table = collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return TLBFlushRateGraph(
                collection_data=collection_data,
                perf_table=perf_table
            )
        return None
