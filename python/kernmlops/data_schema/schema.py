# Abstract definition of CollectionTable and logical collection

from pathlib import Path
from typing import Mapping, cast

import plotext as plt
import polars as pl
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

    def get[T: CollectionTable](self, table_type: type[T]) -> T | None:
        table = self.tables.get(table_type.name(), None)
        if table:
            return cast(T, table)
        return None

    def graph(self, out_dir: Path | None = None) -> None:
        # TODO(Patrick) use verbosity for filtering graphs
        graph_dir = out_dir / self.benchmark / self.id if out_dir else None
        if graph_dir:
            graph_dir.mkdir(parents=True, exist_ok=True)
        for _, collection_table in self.tables.items():
            for graph_type in collection_table.graphs():
                graph = graph_type.with_collection(collection_data=self)
                if not graph:
                    continue
                plt.title(graph.name())
                plt.xlabel(graph.x_axis())
                plt.ylabel(graph.y_axis())
                graph.plot()
                graph.plot_trends()
                plt.show()
                if graph_dir:
                    plt.save_fig(
                        str(graph_dir / f"{graph.base_name().replace(' ', '_').lower()}.plt"),
                        keep_colors=True,
                    )
                plt.clear_figure()

    def dump(self):
        self.graph()
        for name, table in self.tables.items():
            print(f"{name}: {table.table}")

    @classmethod
    def from_tables(
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
