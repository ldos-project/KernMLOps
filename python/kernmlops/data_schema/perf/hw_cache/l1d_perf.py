from __future__ import annotations

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


class L1DPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "l1d_misses"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HW_CACHE

    @classmethod
    def ev_config(cls) -> int:
      return PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_L1D,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      )

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

    @classmethod
    def component_name(cls) -> str:
        return "L1d"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Misses"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> L1DPerfTable:
        return L1DPerfTable(table=table.cast(cls.schema(), strict=True))

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [L1DRateGraph, L1DCumulativeGraph]


class L1DRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return L1DPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return L1DRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class L1DCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return L1DPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return L1DCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None
