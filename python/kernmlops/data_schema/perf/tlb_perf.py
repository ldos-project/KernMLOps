import polars as pl
from bcc import PerfType
from data_schema.memory_usage import MemoryUsageGraph
from data_schema.perf.perf_schema import (
    CumulativePerfGraph,
    CustomHWEventID,
    PerfCollectionTable,
    PerfHWCacheConfig,
    RatePerfGraph,
)
from data_schema.schema import (
    CollectionGraph,
    GraphEngine,
)


class DTLBPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "dtlb_misses"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HW_CACHE

    @classmethod
    def ev_config(cls) -> int:
      return PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_DTLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      )

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

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
        return [DTLBRateGraph, DTLBCumulativeGraph]


class DTLBRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return DTLBPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return DTLBRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class DTLBCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return DTLBPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return DTLBCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class ITLBPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "itlb_misses"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HW_CACHE

    @classmethod
    def ev_config(cls) -> int:
      return PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_ITLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      )

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

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
        return [ITLBRateGraph, ITLBCumulativeGraph]


class ITLBRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return ITLBPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return ITLBRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class ITLBCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return ITLBPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return ITLBCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class TLBFlushPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "tlb_flushes"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.RAW

    @classmethod
    def ev_config(cls) -> int:
        return 0

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return [
            CustomHWEventID(name="TLB_FLUSHES", umask="All"),
            CustomHWEventID(name="TLB_FLUSH", umask="STLB_ANY"),
        ]

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
        return [TLBFlushRateGraph, TLBFlushCumulativeGraph]


class TLBFlushRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return TLBFlushPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return TLBFlushRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class TLBFlushCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return TLBFlushPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return TLBFlushCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class DTLBWalkDurationPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "dtlb_walk_duration"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.RAW

    @classmethod
    def ev_config(cls) -> int:
        return 0

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return [
            CustomHWEventID(name="DTLB_LOAD_MISSES", umask="WALK_DURATION"),
        ]

    @classmethod
    def component_name(cls) -> str:
        return "dTLB"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Walk Durations"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "DTLBWalkDurationPerfTable":
        return DTLBWalkDurationPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []
