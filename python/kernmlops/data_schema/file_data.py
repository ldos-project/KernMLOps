

# from matplotlib import pyplot as plt
import polars as pl
from data_schema.schema import (
    UPTIME_TIMESTAMP,
    CollectionGraph,
    CollectionTable,
)


class FileDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "file_data"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "pid": pl.Int64(),
            "tgid": pl.Int64(),
            UPTIME_TIMESTAMP: pl.Int64(),
            "file_inode": pl.Int64(),
            "file_size_bytes": pl.Int64(),
            "file_name": pl.String(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "FileDataTable":
        return FileDataTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []

    def total_files_opened(self) -> int:
        """Returns the total amount of unique files opened across all cpus."""
        return len(self.filtered_table().select("file_inode").unique("file_inode"))

    def get_file_data(self, filename: str) -> pl.DataFrame:
        return self.filtered_table().filter(pl.col("file_name") == filename)

    def get_first_occurrence_us(self, filename: str) -> int | None:
        file_data = self.get_file_data(filename)
        if len(file_data) == 0:
            return None
        return file_data.sort(
            UPTIME_TIMESTAMP, descending=False
        ).head(n=1).select(
            UPTIME_TIMESTAMP
        ).to_series().to_list()[0]

    def get_last_occurrence_us(self, filename: str) -> int | None:
        file_data = self.get_file_data(filename)
        if len(file_data) == 0:
            return None
        return file_data.sort(
            UPTIME_TIMESTAMP, descending=True
        ).head(n=1).select(
            UPTIME_TIMESTAMP
        ).to_series().to_list()[0]
