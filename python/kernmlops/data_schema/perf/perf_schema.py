from dataclasses import dataclass
from enum import Enum
from typing import Protocol, override

import polars as pl
from data_schema.schema import (
  UPTIME_TIMESTAMP,
  CollectionData,
  CollectionGraph,
  CollectionTable,
  GraphEngine,
  collection_id_column,
  cumulative_pma_as_cdf,
  cumulative_pma_as_pdf,
)


@dataclass(frozen=True)
class CustomHWEventID:
  name: str
  umask: str | None


class PerfHWCacheConfig:
  # From perf_hw_cache_id in uapi/linux/perf_event.h
  class Cache(Enum):
    PERF_COUNT_HW_CACHE_L1D = 0
    PERF_COUNT_HW_CACHE_L1I = 1
    PERF_COUNT_HW_CACHE_LL = 2
    PERF_COUNT_HW_CACHE_DTLB = 3
    PERF_COUNT_HW_CACHE_ITLB = 4
    PERF_COUNT_HW_CACHE_BPU = 5
    PERF_COUNT_HW_CACHE_NODE = 6

  # From perf_hw_cache_op_id in uapi/linux/perf_event.h
  class Op(Enum):
    PERF_COUNT_HW_CACHE_OP_READ = 0
    PERF_COUNT_HW_CACHE_OP_WRITE = 1
    PERF_COUNT_HW_CACHE_OP_PREFETCH = 2

  # From perf_hw_cache_op_result_id  in uapi/linux/perf_event.h
  class Result(Enum):
    PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
    PERF_COUNT_HW_CACHE_RESULT_MISS = 1

  # From https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  @classmethod
  def config(cls, cache: Cache, op: Op, result: Result) -> int:
    return (cache.value) | (op.value << 8) | (result.value << 16)


class PerfCollectionTable(CollectionTable, Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def ev_type(cls) -> int: ...

    @classmethod
    def ev_config(cls) -> int: ...

    @classmethod
    def hw_ids(cls) -> list[CustomHWEventID]: ...

    @classmethod
    def cumulative_column_name(cls) -> str:
        return f"cumulative_{cls.name()}"

    @classmethod
    def component_name(cls) -> str:
        """Name of the component being measured, ex. iTLB"""
        ...

    @classmethod
    def measured_event_name(cls) -> str:
        """Type of event being measured, ex. Misses"""
        ...

    @override
    @classmethod
    def from_df_id(cls, table: pl.DataFrame, collection_id: str) -> "CollectionTable":
        return cls.from_df(
            table=table.with_columns(
                pl.lit(collection_id).alias(collection_id_column())
            ).rename({
                "cumulative_count": cls.cumulative_column_name(),
            })
        )

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "pid": pl.Int64(),
            "tgid": pl.Int64(),
            UPTIME_TIMESTAMP: pl.Int64(),
            "collection_id": pl.String(),
            cls.cumulative_column_name(): pl.Int64(),
            "pmu_enabled_time_us": pl.Int64(),
            "pmu_running_time_us": pl.Int64(),
        })

    def total_cumulative(self) -> int:
        return self.filtered_table().group_by("cpu").max().sum().select(
            self.cumulative_column_name()
        ).to_series().to_list()[0]

    # the raw data is a cumulative representation, this returns the deltas
    def as_pdf(self) -> pl.DataFrame:
        return cumulative_pma_as_pdf(
            self.filtered_table(),
            counter_column=self.cumulative_column_name(),
            counter_column_rename=self.name(),
        )

    # the raw data is a cumulative representation, this scales the counts by record time
    def as_cdf(self) -> pl.DataFrame:
        return cumulative_pma_as_cdf(
            self.filtered_table(),
            counter_column=self.cumulative_column_name(),
            counter_column_rename=self.name(),
        )


class RatePerfGraph(CollectionGraph, Protocol):

    graph_engine: GraphEngine
    _perf_table: PerfCollectionTable

    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]: ...

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        """Returns a graph to use for trend lines."""
        return None

    @classmethod
    def base_name(cls) -> str:
        return f"{cls.perf_table_type().component_name()} Performance"

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def __init__(
        self,
        graph_engine: GraphEngine,
        perf_table: PerfCollectionTable,
    ):
        self.graph_engine = graph_engine
        self._perf_table = perf_table

    @property
    def collection_data(self) -> CollectionData:
        return self.graph_engine.collection_data

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return f"{self._perf_table.component_name()} {self._perf_table.measured_event_name()}/msec"

    def plot(self) -> None:
        pdf_df = self._perf_table.as_pdf()
        start_uptime_sec = self.collection_data.start_uptime_sec
        print(f"Total {self._perf_table.component_name()} {self._perf_table.measured_event_name()}: {self._perf_table.total_cumulative()}")

        # group by and plot by cpu
        def plot_rate(pdf_df: pl.DataFrame) -> None:
            pdf_df_by_cpu = pdf_df.group_by("cpu")
            for cpu, pdf_df_group in pdf_df_by_cpu:
                self.graph_engine.plot(
                    (
                        (pdf_df_group.select(UPTIME_TIMESTAMP) / 1_000_000.0) - start_uptime_sec
                    ).to_series().to_list(),
                    (
                        pdf_df_group.select(self._perf_table.name()) / (
                            pdf_df_group.select("span_duration_us") / 1_000.0
                        )
                    ).to_series().to_list(),
                    label=f"CPU {cpu[0]}",
                )
        plot_rate(pdf_df)

    def plot_trends(self) -> None:
        trend_graph_type = self.trend_graph()
        if trend_graph_type is not None:
            trend_graph = trend_graph_type.with_graph_engine(self.graph_engine)
            if trend_graph is not None:
                trend_graph.plot_trends()


class CumulativePerfGraph(CollectionGraph, Protocol):

    graph_engine: GraphEngine
    _perf_table: PerfCollectionTable

    @classmethod
    def perf_table_type(cls) -> type[PerfCollectionTable]: ...

    @classmethod
    def trend_graph(cls) -> type[CollectionGraph] | None:
        """Returns a graph to use for trend lines."""
        return None

    @classmethod
    def base_name(cls) -> str:
        return f"{cls.perf_table_type().component_name()} Cumulative"

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def __init__(
        self,
        graph_engine: GraphEngine,
        perf_table: PerfCollectionTable,
    ):
        self.graph_engine = graph_engine
        self._perf_table = perf_table

    @property
    def collection_data(self) -> CollectionData:
        return self.graph_engine.collection_data

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return f"{self._perf_table.component_name()} {self._perf_table.measured_event_name()}"

    def plot(self) -> None:
        cdf_df = self._perf_table.as_cdf()
        start_uptime_sec = self.collection_data.start_uptime_sec

        # group by and plot by cpu
        def plot_cumulative(cdf_df: pl.DataFrame) -> None:
            cdf_df_by_cpu = cdf_df.group_by("cpu")
            for cpu, cdf_df_group in cdf_df_by_cpu:
                self.graph_engine.plot(
                    (
                        (cdf_df_group.select(UPTIME_TIMESTAMP) / 1_000_000.0) - start_uptime_sec
                    ).to_series().to_list(),
                    (
                        cdf_df_group.select(self._perf_table.name())
                    ).to_series().to_list(),
                    label=f"CPU {cpu[0]}",
                )
        plot_cumulative(cdf_df)

    def plot_trends(self) -> None:
        pass
