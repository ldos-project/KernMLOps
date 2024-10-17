# Abstract definition of CollectionTable and logical collection

import json
from pathlib import Path
from typing import Mapping, cast

import plotext
import polars as pl
from matplotlib import pyplot
from typing_extensions import Protocol


def collection_id_column() -> str:
    return "collection_id"


def _type_map(table_types: list[type["CollectionTable"]]) -> Mapping[str, type["CollectionTable"]]:
    return {
        table_type.name(): table_type
        for table_type in table_types
    }


class CollectionGraph(Protocol):

    @classmethod
    def with_collection(cls, collection_data: "CollectionData") -> "CollectionGraph | None": ...

    @classmethod
    def base_name(cls) -> str: ...

    def name(self) -> str: ...

    def x_axis(self) -> str: ...

    def y_axis(self) -> str: ...

    def plot(self) -> None: ...

    def plot_trends(self) -> None: ...


class CollectionTable(Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def schema(cls) -> pl.Schema: ...

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CollectionTable": ...

    @classmethod
    def from_df_id(cls, table: pl.DataFrame, collection_id: str) -> "CollectionTable":
        return cls.from_df(
            table=table.with_columns(pl.lit(collection_id).alias(collection_id_column()))
        )

    @property
    def table(self) -> pl.DataFrame: ...

    def filtered_table(self) -> pl.DataFrame:
        # Best effort filter to remove invalid data points
        ...

    def graphs(self) -> list[type[CollectionGraph]]: ...


# TODO(Patrick): Add simple class for holding tables from multiple collections
class CollectionsTable[T: CollectionTable]:
    pass


class SystemInfoTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "system_info"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "SystemInfoTable":
        return SystemInfoTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table
        # TODO(Patrick): proper error handling
        assert len(self.table) == 1

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []

    @property
    def id(self) -> str:
        return self.table[
            collection_id_column()
        ][0]

    @property
    def pid(self) -> int:
        return self.table[
            "collection_pid"
        ][0]

    @property
    def benchmark(self) -> str:
        return self.table[
            "benchmark_name"
        ][0]

    @property
    def start_uptime_sec(self) -> int:
        return self.table[
            "uptime_sec"
        ][0]

    @property
    def benchmark_time_sec(self) -> int:
        return self.table[
            "collection_time_sec"
        ][0]

    @property
    def cpus(self) -> int:
        return self.table[
            "cores"
        ][0]


class CollectionData:

    def __init__(self, collection_tables: Mapping[str, CollectionTable]):
        self._tables = collection_tables
        system_info = self.get(SystemInfoTable)
        # TODO(Patrick): Add proper error handling
        assert isinstance(system_info, SystemInfoTable)
        assert len(system_info.table) == 1
        self._system_info = system_info
        self._plt = plotext

    @property
    def tables(self) -> Mapping[str, CollectionTable]:
        return self._tables

    @property
    def system_info(self) -> SystemInfoTable:
        return self._system_info

    @property
    def id(self) -> str:
        return self.system_info.id

    @property
    def pid(self) -> int:
        return self.system_info.pid

    @property
    def benchmark(self) -> str:
        return self.system_info.benchmark

    @property
    def start_uptime_sec(self) -> int:
        return self.system_info.start_uptime_sec

    @property
    def benchmark_time_sec(self) -> int:
        return self.system_info.benchmark_time_sec

    @property
    def cpus(self) -> int:
        return self.system_info.cpus

    @property
    def plt(self):
        return self._plt

    def get[T: CollectionTable](self, table_type: type[T]) -> T | None:
        table = self.tables.get(table_type.name(), None)
        if table:
            return cast(T, table)
        return None

    def graph(self, out_dir: Path | None = None, *, use_matplot: bool = False, no_trends: bool = False) -> None:
        from kernmlops_benchmark import benchmarks
        self._plt = pyplot if use_matplot else plotext

        # TODO(Patrick) use verbosity for filtering graphs
        graph_dir = out_dir / self.benchmark / self.id if out_dir else None
        if graph_dir:
            graph_dir.mkdir(parents=True, exist_ok=True)
        for _, collection_table in self.tables.items():
            for graph_type in collection_table.graphs():
                graph = graph_type.with_collection(collection_data=self)
                if not graph:
                    continue
                figure = None
                if self.plt is pyplot:
                    figure = pyplot.figure(graph.name())

                self.plt.title(graph.name())
                self.plt.xlabel(graph.x_axis())
                self.plt.ylabel(graph.y_axis())
                graph.plot()
                if not no_trends:
                    graph.plot_trends()
                if self.benchmark in benchmarks:
                    benchmarks[self.benchmark].plot_events(collection_data=self)
                if self.plt is pyplot:
                    pyplot.legend(loc="upper left")

                if figure is not None:
                    manager = pyplot.get_current_fig_manager()
                    if manager is not None:
                        manager.full_screen_toggle()
                    figure.show()
                else:
                    self.plt.show()

                if graph_dir:
                    if self.plt is plotext:
                        plotext.save_fig(
                            str(graph_dir / f"{graph.base_name().replace(' ', '_').lower()}.plt"),
                            keep_colors=True,
                        )
                    else:
                        pyplot.savefig(
                            str(graph_dir / f"{graph.base_name().replace(' ', '_').lower()}.png"),
                            dpi=100,
                        )
                if self.plt is plotext:
                    plotext.clear_figure()
        if self.plt is pyplot:
            input()

    def dump(self, *, use_matplot: bool, no_trends: bool = False):
        self.graph(no_trends=no_trends, use_matplot=use_matplot)
        for name, table in self.tables.items():
            if name == SystemInfoTable.name():
                print(f"{name}: {json.dumps(table.table.row(0, named=True), indent=4)}")
            else:
                print(f"{name}: {table.table}")

    @classmethod
    def from_tables(
        cls,
        tables: list[CollectionTable],
    ) -> "CollectionData":
        return CollectionData({
            collection_table.name(): collection_table
            for collection_table in tables
        })

    @classmethod
    def from_dfs(
        cls,
        tables: Mapping[str, pl.DataFrame],
        table_types: list[type[CollectionTable]],
    ) -> "CollectionData":
        collection_tables = dict[str, CollectionTable]()
        type_map = _type_map(table_types)
        for name, table in tables.items():
            collection_tables[name] = type_map[name].from_df(table)
        return CollectionData(collection_tables)

    @classmethod
    def from_data(
        cls,
        data_dir: Path,
        collection_id: str,
        table_types: list[type[CollectionTable]],
    ) -> "CollectionData":
        collection_tables = dict[str, CollectionTable]()
        type_map = _type_map(table_types)
        dataframe_dirs = [
            x for x in data_dir.iterdir()
            if x.is_dir() and x.name in type_map
        ]
        for dataframe_dir in dataframe_dirs:
            dfs = [
                pl.read_parquet(x) for x in dataframe_dir.iterdir()
                if x.is_file() and x.suffix == ".parquet" and x.name.startswith(collection_id)
            ]
            # Throw explainable error
            assert len(dfs) <= 1
            if dfs:
                collection_tables[dataframe_dir.name] = type_map[dataframe_dir.name].from_df(dfs[0])
        return CollectionData(collection_tables)


def cumulative_pma_as_pdf(table: pl.DataFrame, *, counter_column: str, counter_column_rename: str) -> pl.DataFrame:
    cumulative_columns = [
        counter_column,
        "pmu_enabled_time_us",
        "pmu_running_time_us",
    ]
    final_select = [
        column
        for column in table.columns
        if column not in cumulative_columns
    ]
    final_select.extend([counter_column_rename, "span_duration_us"])
    by_cpu_pdf_dfs = [
        by_cpu_df.lazy().sort("ts_uptime_us").with_columns(
            pl.col(counter_column).shift(1, fill_value=0).alias(f"{counter_column}_shifted"),
            pl.col("pmu_running_time_us").shift(1, fill_value=0).alias("pmu_running_time_us_shifted"),
            pl.col("pmu_enabled_time_us").shift(1, fill_value=0).alias("pmu_enabled_time_us_shifted"),
        ).with_columns(
            (pl.col(counter_column) - pl.col(f"{counter_column}_shifted")).alias(f"{counter_column_rename}_raw"),
            (pl.col("pmu_running_time_us") - pl.col("pmu_running_time_us_shifted")).alias("span_duration_us"),
        ).with_columns(
            (
                (
                    pl.col("span_duration_us")
                ) / (
                    pl.col("pmu_enabled_time_us") - pl.col("pmu_enabled_time_us_shifted")
                )
            ).alias("sampling_scaling"),
        ).with_columns(
            (pl.col(f"{counter_column_rename}_raw") * pl.col("sampling_scaling")).alias(counter_column_rename),
        ).select(final_select)
        for _, by_cpu_df in table.group_by("cpu")
    ]
    return pl.concat(by_cpu_pdf_dfs).collect()
