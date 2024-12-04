from __future__ import annotations

import polars as pl
from bcc import PerfType
from data_schema.memory_usage import MemoryUsageGraph
from data_schema.perf.perf_schema import (
    CumulativePerfGraph,
    CustomHWEventID,
    PerfCollectionTable,
    PerfHardwareConfig,
    RatePerfGraph,
)
from data_schema.schema import (
    CollectionGraph,
    GraphEngine,
)


class CPUCyclesPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "cpu_cycles"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HARDWARE

    @classmethod
    def ev_config(cls) -> int:
      return PerfHardwareConfig.config(
          event=PerfHardwareConfig.Event.PERF_COUNT_HW_REF_CPU_CYCLES,
      )

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

    @classmethod
    def component_name(cls) -> str:
        return "cpu_cycles"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Counts"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> CPUCyclesPerfTable:
        return CPUCyclesPerfTable(table=table.cast(cls.schema(), strict=True))

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [CPUCyclesRateGraph, CPUCyclesCumulativeGraph]


class CPUCyclesRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return CPUCyclesPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return CPUCyclesRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class CPUCyclesCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return CPUCyclesPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return CPUCyclesCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class InstructionsPerfTable(PerfCollectionTable):

    @classmethod
    def name(cls) -> str:
        return "num_instructions"

    @classmethod
    def ev_type(cls) -> int:
        return PerfType.HARDWARE

    @classmethod
    def ev_config(cls) -> int:
      return PerfHardwareConfig.config(
          event=PerfHardwareConfig.Event.PERF_COUNT_HW_INSTRUCTIONS,
      )

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]:
        return []

    @classmethod
    def component_name(cls) -> str:
        return "num_instructions"

    @classmethod
    def measured_event_name(cls) -> str:
        return "Counts"

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> InstructionsPerfTable:
        return InstructionsPerfTable(table=table.cast(cls.schema(), strict=True))

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [InstructionsRateGraph, InstructionsCumulativeGraph]


class InstructionsRateGraph(RatePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return InstructionsPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return MemoryUsageGraph

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return InstructionsRateGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None


class InstructionsCumulativeGraph(CumulativePerfGraph):
    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]:
        return InstructionsPerfTable

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        return None

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        perf_table = graph_engine.collection_data.get(cls.perf_table_type())
        if perf_table is not None:
            return InstructionsCumulativeGraph(
                graph_engine=graph_engine,
                perf_table=perf_table
            )
        return None
