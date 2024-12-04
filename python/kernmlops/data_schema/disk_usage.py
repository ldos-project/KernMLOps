from __future__ import annotations

import polars as pl
from data_schema.schema import (
    UPTIME_TIMESTAMP,
    CollectionGraph,
    CollectionTable,
    GraphEngine,
)


class DiskUsageTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "disk_usage"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            UPTIME_TIMESTAMP: pl.Int64(),

            "reads_kb": pl.Int64(),
            "write_kb": pl.Int64(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> DiskUsageTable:
        return DiskUsageTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [DiskUsageGraph]


class DiskUsageGraph(CollectionGraph):

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        disk_usage_table = graph_engine.collection_data.get(DiskUsageTable)
        if disk_usage_table is not None:
            return DiskUsageGraph(
                graph_engine=graph_engine,
                disk_usage_table=disk_usage_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "System Disk Usage"

    @property
    def plot_lines(self) -> list[str]:
        return [
            "read_kb",
            "write_kb",
        ]

    def __init__(
        self,
        graph_engine: GraphEngine,
        disk_usage_table: DiskUsageTable,
    ):
        self.graph_engine = graph_engine
        self.collection_data = graph_engine.collection_data
        self._disk_usage_table = disk_usage_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Memory (KB)"

    def plot(self) -> None:
        pass

    def plot_trends(self) -> None:
        disk_df = self._disk_usage_table.filtered_table()
        start_uptime_sec = self.collection_data.start_uptime_sec

        for plot_line in self.plot_lines:
            self.graph_engine.plot(
                (
                    (disk_df.select(UPTIME_TIMESTAMP) / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (disk_df.select(plot_line)).to_series().to_list(),
                label=plot_line.replace("bytes", "kb"),
                y_axis=self.y_axis(),
            )
