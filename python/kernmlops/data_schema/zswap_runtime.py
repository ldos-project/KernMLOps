import polars as pl
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)


class ZswapRuntimeDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "zswap_runtime"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "ZswapRuntimeDataTable":
        return ZswapRuntimeDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []

    def by_pid(self, pids: int | list[int]) -> pl.DataFrame:
        if isinstance(pids, int):
            pids = [pids]
        return self.filtered_table().filter(pl.col("pid").is_in(pids))
