"""Final YCSB exports, kept separate from invocation execution metadata."""

from typing import Self

import polars as pl
from data_schema.schema import CollectionGraph, CollectionTable


class _YCSBTable(CollectionTable):
    @classmethod
    def from_df(cls, table: pl.DataFrame) -> Self:
        return cls(table=table)

    def __init__(self, table: pl.DataFrame):
        schema = self.schema()
        self._table = table.select(schema.names()).cast(
            {name: dtype for name, dtype in schema.items()}, strict=True
        )

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        # Failed invocations, null values, and repeated measurements are evidence.
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []


class YCSBInvocationsTable(_YCSBTable):
    @classmethod
    def name(cls) -> str:
        return "ycsb_invocations"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema(
            {
                "collection_id": pl.String,
                "invocation_id": pl.UInt32,
                "benchmark_name": pl.String,
                "run_label": pl.String,
                "phase": pl.String,
                "iteration": pl.UInt32,
                "outer_iteration": pl.UInt32,
                "argv": pl.List(pl.String),
                "start_monotonic_ns": pl.UInt64,
                "end_monotonic_ns": pl.UInt64,
                "exit_code": pl.Int32,
                "execution_state": pl.String,
                "parse_state": pl.String,
                "error": pl.String,
                "stdout_path": pl.String,
                "stderr_path": pl.String,
            }
        )


class YCSBMetricsTable(_YCSBTable):
    @classmethod
    def name(cls) -> str:
        return "ycsb_metrics"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema(
            {
                "collection_id": pl.String,
                "invocation_id": pl.UInt32,
                "source": pl.String,
                "line_number": pl.UInt64,
                "section": pl.String,
                "metric": pl.String,
                "value_raw": pl.String,
                "value": pl.Float64,
                "unit": pl.String,
            }
        )
