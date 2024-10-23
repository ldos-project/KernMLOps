import json
from typing import Mapping

import polars as pl

from data_schema.process_metadata import ProcessMetadataTable
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
    GraphEngine,
)


class QuantaRuntimeTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "quanta_runtime"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "pid": pl.Int64(),
            "tgid": pl.Int64(),
            "quanta_end_uptime_us": pl.Int64(),
            "quanta_run_length_us": pl.Int64(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "QuantaRuntimeTable":
        return QuantaRuntimeTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        # filter out invalid data points due to data loss
        # these are probably due to hardware threads not running
        initial_datapoints = len(self.table)
        max_run_length = 60_000
        quanta_df = self.table.filter(
          pl.col("quanta_run_length_us") < max_run_length
        )
        datapoints_removed = initial_datapoints - len(quanta_df)
        # TODO(Patrick): use logging
        print(f"Filtered out {datapoints_removed} datapoints with max run length {max_run_length}us")
        return quanta_df

    def graphs(self) -> list[type[CollectionGraph]]:
        return [QuantaRuntimeGraph]

    def total_runtime_us(self) -> int:
        """Returns the total amount of runtime recorded across all cpus."""
        return self.filtered_table().select(
            "quanta_run_length_us"
        ).sum()["quanta_run_length_us"].to_list()[0]

    def per_cpu_total_runtime_sec(self) -> pl.DataFrame:
        """Returns the total amount of runtime recorded per cpu."""
        return self.filtered_table().group_by(
            "cpu"
        ).agg(
            pl.sum("quanta_run_length_us")
        ).select([
            "cpu",
            (pl.col("quanta_run_length_us") / 1_000_000.0).alias("cpu_total_runtime_sec"),
        ]).sort("cpu_total_runtime_sec")

    def top_k_runtime(self, k: int) -> pl.DataFrame:
        """Returns the pids and execution time of the k processes with the most execution time."""
        # in kernel space thread id and pid meanings are swapped
        return self.filtered_table().select(
            [pl.col("tgid").alias("pid"), "quanta_run_length_us"]
        ).group_by(
            "pid"
        ).sum().sort(
            "quanta_run_length_us", descending=True
        ).limit(k)


class QuantaQueuedTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "quanta_queued_time"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "pid": pl.Int64(),
            "tgid": pl.Int64(),
            "quanta_end_uptime_us": pl.Int64(),
            "quanta_queued_time_us": pl.Int64(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "QuantaQueuedTable":
        return QuantaQueuedTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        # filter out outliers
        initial_datapoints = len(self.table)
        max_queue_time_us = 1_000_000
        quanta_df = self.table.filter(
          pl.col("quanta_queued_time_us") < max_queue_time_us
        )
        datapoints_removed = initial_datapoints - len(quanta_df)
        # TODO(Patrick): use logging
        print(f"Filtered out {datapoints_removed} datapoints with max queue time {max_queue_time_us}us")
        return quanta_df

    def graphs(self) -> list[type[CollectionGraph]]:
        return [QuantaQueuedGraph]

    def total_queued_time_us(self) -> int:
        """Returns the total amount of queued time recorded across all cpus."""
        return self.filtered_table().select(
            "quanta_queued_time_us"
        ).sum()["quanta_queued_time_us"].to_list()[0]

    def per_cpu_total_runtime_sec(self) -> pl.DataFrame:
        """Returns the total amount of Queued time recorded per cpu."""
        return self.filtered_table().group_by(
            "cpu"
        ).agg(
            pl.sum("quanta_queued_time_us")
        ).select([
            "cpu",
            (pl.col("quanta_queued_time_us") / 1_000_000.0).alias("cpu_total_queued_time_sec"),
        ]).sort("cpu_total_queued_time_sec")

    def top_k_queued_time(self, k: int) -> pl.DataFrame:
        """Returns the pids and execution time of the k processes with the most queued time."""
        # in kernel space thread id and pid meanings are swapped
        return self.filtered_table().select(
            [pl.col("tgid").alias("pid"), "quanta_queued_time_us"]
        ).group_by(
            "pid"
        ).sum().sort(
            "quanta_queued_time_us", descending=True
        ).limit(k)


class QuantaRuntimeGraph(CollectionGraph):

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        quanta_table = graph_engine.collection_data.get(QuantaRuntimeTable)
        if quanta_table is not None:
            return QuantaRuntimeGraph(
                graph_engine=graph_engine,
                quanta_table=quanta_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "Quanta Runtimes"

    def __init__(
        self,
        graph_engine: GraphEngine,
        quanta_table: QuantaRuntimeTable,
    ):
        self.graph_engine = graph_engine
        self.collection_data = self.graph_engine.collection_data
        self._quanta_table = quanta_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Quanta Run Length (msec)"

    def plot(self) -> None:
        quanta_df = self._quanta_table.filtered_table()
        start_uptime_sec = self.collection_data.start_uptime_sec

        # group by and plot by cpu
        quanta_df_by_cpu = quanta_df.group_by("cpu")
        for cpu, quanta_df_group in quanta_df_by_cpu:
            self.graph_engine.scatter(
                (
                    (quanta_df_group.select("quanta_end_uptime_us") / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (quanta_df_group.select("quanta_run_length_us") / 1_000.0).to_series().to_list(),
                label=f"CPU {cpu[0]}",
            )

    def plot_trends(self) -> None:
        quanta_table = self._quanta_table
        quanta_df = quanta_table.filtered_table()
        start_uptime_sec = self.collection_data.start_uptime_sec
        collector_pid = self.collection_data.pid
        top_k = quanta_table.top_k_runtime(k=3)
        print(top_k)
        pid_labels: Mapping[int, str] = self._get_pid_labels(top_k["pid"].to_list() + [collector_pid], collector_pid)
        print(json.dumps(pid_labels, indent=4))
        for pid, label in pid_labels.items():
            # Add trend of collector process to graph
            collector_runtimes = quanta_df.filter(
                pl.col("tgid") == pid
            )
            self.graph_engine.plot(
                (
                    (collector_runtimes.select("quanta_end_uptime_us") / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (collector_runtimes.select("quanta_run_length_us") / 1_000.0).to_series().to_list(),
                label="Collector Process" if collector_pid == pid else label[:35],
            )
        print(f"Total processor time per cpu:\n{quanta_table.per_cpu_total_runtime_sec()}")

    def _get_pid_labels(self, pids: list[int], collector_pid: int | None = None) -> Mapping[int, str]:
        process_table = self.collection_data.get(ProcessMetadataTable)
        if not process_table:
            return {
                pid: "Collector Process" if collector_pid == pid else f"PID: {pid}"
                for pid in pids
            }
        assert process_table is not None
        process_data = process_table.by_pid(pids)
        # TODO(Patrick): extract process-specific important args like file to compile for cc1
        process_pid_map = {
            pid: f"{name} {(cmdline + ' ').split(' ', maxsplit=1)[1]}"
            for pid, name, cmdline in zip(
                process_data["pid"].to_list(),
                process_data["name"].to_list(),
                process_data["cmdline"].to_list(),
                strict=True
            )
        }
        return {
            pid: process_pid_map[pid] if pid in process_pid_map else f"PID: {pid}"
            for pid in pids
        }


class QuantaQueuedGraph(CollectionGraph):

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        quanta_table = graph_engine.collection_data.get(QuantaQueuedTable)
        if quanta_table is not None:
            return QuantaQueuedGraph(
                graph_engine=graph_engine,
                quanta_table=quanta_table,
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "Quanta Queued Time"

    def __init__(
        self,
        graph_engine: GraphEngine,
        quanta_table: QuantaQueuedTable,
    ):
        self.graph_engine = graph_engine
        self.collection_data = graph_engine.collection_data
        self._quanta_table = quanta_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Quanta Queued Time (msec)"

    def plot(self) -> None:
        quanta_df = self._quanta_table.filtered_table()
        start_uptime_sec = self.collection_data.start_uptime_sec

        # group by and plot by cpu
        quanta_df_by_cpu = quanta_df.group_by("cpu")
        for cpu, quanta_df_group in quanta_df_by_cpu:
            self.graph_engine.scatter(
                (
                    (quanta_df_group.select("quanta_end_uptime_us") / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (quanta_df_group.select("quanta_queued_time_us") / 1_000.0).to_series().to_list(),
                label=f"CPU {cpu[0]}",
            )

    def plot_trends(self) -> None:
        quanta_table = self._quanta_table
        quanta_df = quanta_table.filtered_table()
        start_uptime_sec = self.collection_data.start_uptime_sec
        collector_pid = self.collection_data.pid
        top_k = quanta_table.top_k_queued_time(k=3)
        print(top_k)
        pid_labels: Mapping[int, str] = self._get_pid_labels(top_k["pid"].to_list() + [collector_pid], collector_pid)
        print(json.dumps(pid_labels, indent=4))
        for pid, label in pid_labels.items():
            # Add trend of collector process to graph
            collector_runtimes = quanta_df.filter(
                pl.col("tgid") == pid
            )
            self.graph_engine.plot(
                (
                    (collector_runtimes.select("quanta_end_uptime_us") / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (collector_runtimes.select("quanta_queued_time_us") / 1_000.0).to_series().to_list(),
                label="Collector Process" if collector_pid == pid else label[:35],
            )
        print(f"Total processor time per cpu:\n{quanta_table.per_cpu_total_runtime_sec()}")

    def _get_pid_labels(self, pids: list[int], collector_pid: int | None = None) -> Mapping[int, str]:
        process_table = self.collection_data.get(ProcessMetadataTable)
        if not process_table:
            return {
                pid: "Collector Process" if collector_pid == pid else f"PID: {pid}"
                for pid in pids
            }
        assert process_table is not None
        process_data = process_table.by_pid(pids)
        # TODO(Patrick): extract process-specific important args like file to compile for cc1
        process_pid_map = {
            pid: f"{name} {(cmdline + ' ').split(' ', maxsplit=1)[1]}"
            for pid, name, cmdline in zip(
                process_data["pid"].to_list(),
                process_data["name"].to_list(),
                process_data["cmdline"].to_list(),
                strict=True
            )
        }
        return {
            pid: process_pid_map[pid] if pid in process_pid_map else f"PID: {pid}"
            for pid in pids
        }
