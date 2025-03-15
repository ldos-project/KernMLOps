import polars as pl
from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)


class CBMMEagerDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "cbmm_eager"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CBMMEagerDataTable":
        return CBMMEagerDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []

class CBMMPrezeroingDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "cbmm_prezero"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "CBMMPrezeroingDataTable":
        return CBMMPrezeroingDataTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []
