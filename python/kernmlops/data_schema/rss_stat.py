import polars as pl
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)


class TraceMMRSSStatDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "mm_rss_stat"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TraceMMRSSStatDataTable":
        return TraceMMRSSStatDataTable(table=table)

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
