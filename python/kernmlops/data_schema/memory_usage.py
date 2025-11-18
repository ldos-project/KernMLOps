import polars as pl
from data_schema.schema import (
    UPTIME_TIMESTAMP,
    CollectionGraph,
    CollectionTable,
    GraphEngine,
)


class MemoryUsageTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "memory_usage"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            UPTIME_TIMESTAMP: pl.Int64(),

            "mem_total_bytes": pl.Int64(),
            "mem_free_bytes": pl.Int64(),
            "mem_available_bytes": pl.Int64(),
            "buffers_bytes": pl.Int64(),
            "cached_bytes": pl.Int64(),

            "swap_total_bytes": pl.Int64(),
            "swap_free_bytes": pl.Int64(),

            # data waiting to be written back to disk
            "dirty_bytes": pl.Int64(),
            "writeback_bytes": pl.Int64(),

            "anon_pages_total_bytes": pl.Int64(),
            "anon_hugepages_total_bytes": pl.Int64(),
            "mapped_total_bytes": pl.Int64(),
            "shmem_total_bytes": pl.Int64(),

            "hugepages_total": pl.Int64(),
            "hugepages_free": pl.Int64(),
            "hugepages_reserved": pl.Int64(),
            "hugepage_size_bytes": pl.Int64(),

            "hardware_corrupted_bytes": pl.Int64(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "MemoryUsageTable":
        return MemoryUsageTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [MemoryUsageGraph]


class MemoryUsageGraph(CollectionGraph):

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        memory_usage_table = graph_engine.collection_data.get(MemoryUsageTable)
        if memory_usage_table is not None:
            return MemoryUsageGraph(
                graph_engine=graph_engine,
                memory_usage_table=memory_usage_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "System Memory Usage"

    @property
    def plot_lines(self) -> list[str]:
        return [
            "cached_bytes",
            "anon_pages_total_bytes",
            "anon_hugepages_total_bytes",
            "mapped_total_bytes",
            "shmem_total_bytes",
        ]

    def __init__(
        self,
        graph_engine: GraphEngine,
        memory_usage_table: MemoryUsageTable,
    ):
        self.graph_engine = graph_engine
        self.collection_data = graph_engine.collection_data
        self._memory_usage_table = memory_usage_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Memory (GB)"

    def plot(self) -> None:
        pass

    def plot_trends(self) -> None:
        memory_df = self._memory_usage_table.filtered_table()

        for plot_line in self.plot_lines:
            self.graph_engine.plot(
                self.collection_data.normalize_uptime_sec(memory_df),
                (memory_df.select(plot_line) / (1_024.0)**3).to_series().to_list(),
                label=plot_line.replace("bytes", "gb"),
                y_axis=self.y_axis(),
            )
