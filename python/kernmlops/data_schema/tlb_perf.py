import polars as pl

from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    CollectionTable,
    cumulative_pma_as_pdf,
)


class TLBPerfTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "tlb_perf"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "dtlb_event": pl.Boolean(),
            "itlb_event": pl.Boolean(),
            "cpu": pl.Int64(),
            "ts_uptime_us": pl.Int64(),
            "collection_id": pl.String(),
            "cumulative_tlb_misses": pl.Int64(),
            "pmu_enabled_time_us": pl.Int64(),
            "pmu_running_time_us": pl.Int64(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TLBPerfTable":
        return TLBPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [DTLBPerfGraph, ITLBPerfGraph]

    def dtlb_table(self) -> pl.DataFrame:
        return self.filtered_table().filter(
            pl.col("dtlb_event")
        )

    def itlb_table(self) -> pl.DataFrame:
        return self.filtered_table().filter(
            pl.col("itlb_event")
        )

    def _total_misses(self, table: pl.DataFrame) -> int:
        return table.group_by("cpu").max().sum().select("cumulative_tlb_misses").to_series().to_list()[0]

    def total_dtlb_misses(self) -> int:
        return self._total_misses(self.dtlb_table())

    def total_itlb_misses(self) -> int:
        return self._total_misses(self.itlb_table())

    # the raw data is a cumulative representation, this returns the deltas
    def as_dtlb_pdf(self) -> pl.DataFrame:
        dtlb_df = self.dtlb_table()
        return cumulative_pma_as_pdf(
            dtlb_df,
            counter_column="cumulative_tlb_misses",
            counter_column_rename="tlb_misses",
        )

    def as_itlb_pdf(self) -> pl.DataFrame:
        itlb_df = self.itlb_table()
        return cumulative_pma_as_pdf(
            itlb_df,
            counter_column="cumulative_tlb_misses",
            counter_column_rename="tlb_misses",
        )


class DTLBPerfGraph(CollectionGraph):

    @classmethod
    def with_collection(cls, collection_data: "CollectionData") -> "CollectionGraph | None":
        tlb_perf_table = collection_data.get(TLBPerfTable)
        if tlb_perf_table is not None:
            return DTLBPerfGraph(
                collection_data=collection_data,
                tlb_perf_table=tlb_perf_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "dTLB Performance"

    def __init__(
        self,
        collection_data: CollectionData,
        tlb_perf_table: TLBPerfTable,
    ):
        self.collection_data = collection_data
        self._tlb_perf_table = tlb_perf_table
        self.plt = self.collection_data.plt

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "dTLB Misses/msec"

    def plot(self) -> None:
        dtlb_df = self._tlb_perf_table.as_dtlb_pdf()
        # TODO(Patrick): fix tlb misses to scale by span time
        start_uptime_sec = self.collection_data.start_uptime_sec
        print(f"Total dTLB misses: {self._tlb_perf_table.total_dtlb_misses()}")

        # group by and plot by cpu
        def plot_tlb(tlb_df: pl.DataFrame, *, tlb_type: str) -> None:
            tlb_df_by_cpu = tlb_df.group_by("cpu")
            for cpu, tlb_df_group in tlb_df_by_cpu:
                self.plt.plot(
                    (
                        (tlb_df_group.select("ts_uptime_us") / 1_000_000.0) - start_uptime_sec
                    ).to_series().to_list(),
                    (
                        tlb_df_group.select("tlb_misses") / (
                            tlb_df_group.select("span_duration_us") / 1_000.0
                        )
                    ).to_series().to_list(),
                    label=f"{tlb_type} CPU {cpu[0]}",
                )
        plot_tlb(dtlb_df, tlb_type="dTLB")

    def plot_trends(self) -> None:
        pass


class ITLBPerfGraph(CollectionGraph):

    @classmethod
    def with_collection(cls, collection_data: "CollectionData") -> "CollectionGraph | None":
        tlb_perf_table = collection_data.get(TLBPerfTable)
        if tlb_perf_table is not None:
            return ITLBPerfGraph(
                collection_data=collection_data,
                tlb_perf_table=tlb_perf_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "iTLB Performance"

    def __init__(
        self,
        collection_data: CollectionData,
        tlb_perf_table: TLBPerfTable,
    ):
        self.collection_data = collection_data
        self._tlb_perf_table = tlb_perf_table
        self.plt = self.collection_data.plt

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "iTLB Misses/msec"

    def plot(self) -> None:
        itlb_df = self._tlb_perf_table.as_itlb_pdf()
        # TODO(Patrick): fix tlb misses to scale by span time
        start_uptime_sec = self.collection_data.start_uptime_sec
        print(f"Total iTLB misses: {self._tlb_perf_table.total_itlb_misses()}")

        # group by and plot by cpu
        def plot_tlb(tlb_df: pl.DataFrame, *, tlb_type: str) -> None:
            tlb_df_by_cpu = tlb_df.group_by("cpu")
            for cpu, tlb_df_group in tlb_df_by_cpu:
                self.plt.plot(
                    (
                        (tlb_df_group.select("ts_uptime_us") / 1_000_000.0) - start_uptime_sec
                    ).to_series().to_list(),
                    (
                        tlb_df_group.select("tlb_misses") / (
                            tlb_df_group.select("span_duration_us") / 1_000.0
                        )
                    ).to_series().to_list(),
                    label=f"{tlb_type} CPU {cpu[0]}",
                )
        plot_tlb(itlb_df, tlb_type="iTLB")

    def plot_trends(self) -> None:
        pass
