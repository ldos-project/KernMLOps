
import polars as pl
from data_schema.schema import (
    UPTIME_TIMESTAMP,
    CollectionGraph,
    CollectionTable,
    GraphEngine,
)

# from: https://github.com/iovisor/bcc/blob/8d85dcfac86bb7402a20bea5ceba373e5e019b6c/tools/biolatency.py#L328
req_opf = {
    0: "Read",
    1: "Write",
    2: "Flush",
    3: "Discard",
    5: "SecureErase",
    6: "ZoneReset",
    7: "WriteSame",
    9: "WriteZeros"
}
REQ_OP_BITS = 8
REQ_OP_MASK = ((1 << REQ_OP_BITS) - 1)
REQ_SYNC = 1 << (REQ_OP_BITS + 3)
REQ_META = 1 << (REQ_OP_BITS + 4)
REQ_PRIO = 1 << (REQ_OP_BITS + 5)
REQ_NOMERGE = 1 << (REQ_OP_BITS + 6)
REQ_IDLE = 1 << (REQ_OP_BITS + 7)
REQ_FUA = 1 << (REQ_OP_BITS + 9)
REQ_RAHEAD = 1 << (REQ_OP_BITS + 11)
REQ_BACKGROUND = 1 << (REQ_OP_BITS + 12)
REQ_NOWAIT = 1 << (REQ_OP_BITS + 13)
def flags_print(flags: int):
    desc = ""
    # operation
    if flags & REQ_OP_MASK in req_opf:
        desc = req_opf[flags & REQ_OP_MASK]
    else:
        desc = "Unknown"
    # flags
    if flags & REQ_SYNC:
        desc = "Sync-" + desc
    if flags & REQ_META:
        desc = "Metadata-" + desc
    if flags & REQ_FUA:
        desc = "ForcedUnitAccess-" + desc
    if flags & REQ_PRIO:
        desc = "Priority-" + desc
    if flags & REQ_NOMERGE:
        desc = "NoMerge-" + desc
    if flags & REQ_IDLE:
        desc = "Idle-" + desc
    if flags & REQ_RAHEAD:
        desc = "ReadAhead-" + desc
    if flags & REQ_BACKGROUND:
        desc = "Background-" + desc
    if flags & REQ_NOWAIT:
        desc = "NoWait-" + desc
    return desc


class BlockIOQueueTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "block_io_queue_length"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "device": pl.Int64(),
            "sector": pl.Int64(),
            "segments": pl.Int64(),
            "block_io_bytes": pl.Int64(),
            UPTIME_TIMESTAMP: pl.Int64(),
            "block_io_flags": pl.Int64(),
            "queue_length_segment_ios": pl.Int64(),
            "queue_length_4k_ios": pl.Int64(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "BlockIOQueueTable":
        return BlockIOQueueTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []


class BlockIOLatencyTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "block_io_latency"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "device": pl.Int64(),
            "sector": pl.Int64(),
            "segments": pl.Int64(),
            "block_io_bytes": pl.Int64(),
            UPTIME_TIMESTAMP: pl.Int64(),
            "block_latency_us": pl.Int64(),
            "block_io_latency_us": pl.Int64(),
            "block_io_flags": pl.Int64(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "BlockIOLatencyTable":
        return BlockIOLatencyTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []


class BlockIOTable(CollectionTable):
    """Best effort merged table of BlockIOQueueTable and BlockIOLatencyTable."""

    @classmethod
    def name(cls) -> str:
        return "block_io"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "device": pl.Int64(),
            "sector": pl.Int64(),
            "segments": pl.Int64(),
            "block_io_bytes": pl.Int64(),
            UPTIME_TIMESTAMP: pl.Int64(),
            "finish_ts_uptime_us": pl.Int64(),
            "block_latency_us": pl.Int64(),
            "block_io_latency_us": pl.Int64(),
            "measured_latency_us": pl.Int64(),
            "block_io_flags": pl.Int64(),
            "block_io_flags_string": pl.String(),
            "queue_length_segment_ios": pl.Int64(),
            "queue_length_4k_ios": pl.Int64(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "BlockIOTable":
        return BlockIOTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    @classmethod
    def from_tables(cls, queue_table: BlockIOQueueTable, latency_table: BlockIOLatencyTable) -> "BlockIOTable":
        queue_df = queue_table.filtered_table()
        latency_df = latency_table.filtered_table().select([
            "device",
            "sector",
            "segments",
            "block_io_bytes",
            UPTIME_TIMESTAMP,
            "block_latency_us",
            "block_io_latency_us",
            "block_io_flags",
            "collection_id",
        ]).rename({
            UPTIME_TIMESTAMP: "finish_ts_uptime_us",
        })
        block_df = queue_df.join(
            latency_df,
            on=[
                "device",
                "sector",
                "segments",
                "block_io_bytes",
                "block_io_flags",
                "collection_id",
            ],
            how="inner",
        ).with_columns(
            (pl.col("finish_ts_uptime_us") - pl.col(UPTIME_TIMESTAMP)).alias("measured_latency_us"),
            pl.col("block_io_flags").map_elements(
                flags_print, return_dtype=pl.String,
            ).alias("block_io_flags_string"),
        ).filter(
            pl.col("measured_latency_us") > 0
        ).filter(
            (pl.col("measured_latency_us") - 100) < pl.col("block_latency_us")
        ).sort(UPTIME_TIMESTAMP, descending=False)
        return cls.from_df(block_df)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [BlockQueueGraph]

    def summary_df(self) -> pl.DataFrame:
        """Returns a summary of all IO done per device."""
        return self.filtered_table().group_by("device").agg([
            pl.len().alias("disk_io_count"),

            pl.sum("segments").alias("total_segments"),
            (pl.sum("block_io_bytes") / 1024.0 / 1024.0).alias("total_block_io_mb"),
            pl.max("segments").alias("max_segments"),
            pl.max("block_io_bytes").alias("max_block_io_bytes"),

            (pl.sum("block_latency_us") / 1000.0).alias("total_block_latency_ms"),
            (pl.sum("block_io_latency_us") / 1000.0).alias("total_io_block_latency_ms"),
            (pl.sum("measured_latency_us") / 1000.0).alias("total_measured_latency_ms"),
            pl.mean("block_latency_us").alias("avg_block_latency_us"),
            pl.mean("block_io_latency_us").alias("avg_io_block_latency_us"),
            pl.mean("measured_latency_us").alias("avg_measured_latency_us"),
            pl.max("block_latency_us").alias("max_block_latency_us"),
            pl.max("block_io_latency_us").alias("max_io_block_latency_us"),
            pl.max("measured_latency_us").alias("max_measured_latency_us"),

            pl.mean("queue_length_segment_ios").alias("avg_queue_length_segment_ios"),
            pl.mean("queue_length_4k_ios").alias("avg_queue_length_4k_ios"),
            pl.max("queue_length_segment_ios").alias("max_queue_length_segment_ios"),
            pl.max("queue_length_4k_ios").alias("max_queue_length_4k_ios"),
        ])


class BlockQueueGraph(CollectionGraph):

    @classmethod
    def with_graph_engine(cls, graph_engine: GraphEngine) -> CollectionGraph | None:
        block_table = graph_engine.collection_data.get(BlockIOTable)
        if block_table is not None:
            return BlockQueueGraph(
                graph_engine=graph_engine,
                block_table=block_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "Block IO Queue Sizes"

    def __init__(
        self,
        graph_engine: GraphEngine,
        block_table: BlockIOTable,
    ):
        self.graph_engine = graph_engine
        self.collection_data = self.graph_engine.collection_data
        self._block_table = block_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Queue Size"

    def plot(self) -> None:
        block_df = self._block_table.filtered_table()
        with pl.Config(tbl_cols=-1):
            print(self._block_table.summary_df())

        # group by and plot by device
        block_df_by_device = block_df.group_by("device")
        for device, block_df_group in block_df_by_device:
            self.graph_engine.plot(
                self.collection_data.normalize_uptime_sec(block_df_group),
                (block_df_group.select("queue_length_segment_ios")).to_series().to_list(),
                label=f"Device {device[0]} Segment IOs",
            )
            self.graph_engine.plot(
                self.collection_data.normalize_uptime_sec(block_df_group),
                (block_df_group.select("queue_length_4k_ios")).to_series().to_list(),
                label=f"Device {device[0]} 4K IOs",
            )
            self.graph_engine.plot(
                self.collection_data.normalize_uptime_sec(block_df_group),
                (block_df_group.select("block_io_latency_us")).to_series().to_list(),
                label=f"Device {device[0]} Block IO Latency",
                y_axis="IO Latency (us)",
                linestyle="dashed",
            )

    def plot_trends(self) -> None:
        return
