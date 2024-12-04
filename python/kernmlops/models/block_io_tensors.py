import random
from typing import Callable, Mapping

import polars as pl
import torch

file_path_prefix = "data/tensors"

train_df = pl.read_parquet("data/rainsong_curated/block_io/*.parquet").filter(
    pl.col("device").is_in([
        271581184,
        271581185,
        271581186,
    ])
)

test_df = pl.read_parquet("data/rainsong_test_curated/block_io/*.parquet").filter(
    pl.col("device").is_in([
        271581184,
        271581185,
        271581186,
    ])
)


def convert_parquet_to_tensor(data_df: pl.DataFrame, *, transformer: Callable[[list[Mapping[str, int]]], list[int] | None], threshold: float, type: str, suffix: str = "", even: bool = False):
    raw_data = data_df.sort(["device", "ts_uptime_us"]).select([
        "cpu",
        "device",
        "sector",
        "segments",
        "block_io_bytes",
        "ts_uptime_us",
        "block_io_flags",
        "queue_length_segment_ios",
        "queue_length_4k_ios",
        "block_latency_us",
        "collection_id",
    ]).rows(named=True)
    feature_data = list[list[float]]()
    latency_data = list[list[float]]()
    total_fast = 0
    total_slow = 0
    for index in range(len(raw_data) - 3):
        predict_index = index + 3
        predictor_data = raw_data[index:predict_index + 1]
        actual_block_latency = predictor_data[-1]["block_latency_us"]
        fast_io = actual_block_latency < threshold
        slow_io = not fast_io
        if even and fast_io and random.randint(0, 20) < 8:
            continue
        cleaned_predictor_data = transformer(predictor_data)
        if not cleaned_predictor_data:
            continue
        if fast_io:
            total_fast += 1
        if slow_io:
            total_slow += 1

        feature_data.append(cleaned_predictor_data)
        latency_data.append([1 if fast_io else 0, 1 if slow_io else 0])
    features = torch.tensor(feature_data, dtype=torch.float32)
    latencies = torch.tensor(latency_data, dtype=torch.float32)
    even_extension = "reduced_reads." if even else ""
    print(f"Fast IO: {total_fast}")
    print(f"Slow IO: {total_slow}")
    torch.save(features, f"{file_path_prefix}/rainsong_{type}_features.flags.{even_extension}{suffix}tensor")
    torch.save(latencies, f"{file_path_prefix}/rainsong_{type}_latencies_{threshold}.flags.{even_extension}{suffix}tensor")

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

def _explode_flags(flags: int) -> list[int]:
    exploded_flags = list[int]()
    exploded_flags.append(flags & REQ_OP_MASK)
    exploded_flags.append(1 if flags & REQ_SYNC else 0)
    exploded_flags.append(1 if flags & REQ_FUA else 0)
    exploded_flags.append(1 if flags & REQ_PRIO else 0)
    exploded_flags.append(1 if flags & REQ_NOMERGE else 0)
    exploded_flags.append(1 if flags & REQ_IDLE else 0)
    exploded_flags.append(1 if flags & REQ_RAHEAD else 0)
    exploded_flags.append(1 if flags & REQ_BACKGROUND else 0)
    exploded_flags.append(1 if flags & REQ_NOWAIT else 0)
    return exploded_flags # length 9

def _null_data() -> Mapping[str, int]:
    return {
        "cpu": 0,
        "device": 0,
        "sector": 0,
        "segments": 0,
        "block_io_bytes": 0,
        "ts_uptime_us": 0,
        "block_io_flags": 0,
        "queue_length_segment_ios": 0,
        "queue_length_4k_ios": 0,
        "block_latency_us": 0, # consider making this large
        "collection_id": 0,
    }

def _flatten_data(predictor_data: list[Mapping[str, int]]) -> list[int] | None:
    data = list[int]()
    start_ts_us = predictor_data[-1]["ts_uptime_us"]
    device = predictor_data[-1]["device"]
    collection_id = predictor_data[-1]["collection_id"]

    def _append_row(row: Mapping[str, int]):
        data.append(row["cpu"])
        data.append(row["device"])
        data.append(row["sector"])
        data.append(row["segments"])
        data.append(row["block_io_bytes"])
        data.append(-(row["ts_uptime_us"] - start_ts_us) if row["ts_uptime_us"] > 0 else 0)
        data.extend(_explode_flags(row["block_io_flags"]))
        data.append(row["queue_length_segment_ios"])
        data.append(row["queue_length_4k_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < start_ts_us else 0)

    for row in predictor_data:
        if row["device"] == device and row["collection_id"] == collection_id:
            _append_row(row)
        else:
            _append_row(_null_data())

    return data

def _reads_only(predictor_data: list[Mapping[str, int]]) -> list[int] | None:
    data = list[int]()
    start_ts_us = predictor_data[-1]["ts_uptime_us"]
    device = predictor_data[-1]["device"]
    collection_id = predictor_data[-1]["collection_id"]
    exploded_flags = _explode_flags(predictor_data[-1]["block_io_flags"])
    # only include reads
    if exploded_flags[0] != 0:
        return None

    def _append_row(row: Mapping[str, int]):
        data.append(row["cpu"])
        data.append(row["device"])
        data.append(row["sector"])
        data.append(row["segments"])
        data.append(row["block_io_bytes"])
        data.append(-(row["ts_uptime_us"] - start_ts_us) if row["ts_uptime_us"] > 0 else 0)
        data.extend(_explode_flags(row["block_io_flags"]))
        data.append(row["queue_length_segment_ios"])
        data.append(row["queue_length_4k_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < start_ts_us else 0)

    for row in predictor_data:
        if row["device"] == device and row["collection_id"] == collection_id:
            _append_row(row)
        else:
            _append_row(_null_data())

    return data


threshold = 350 #int(test_df.select("block_latency_us").quantile(.95, interpolation="nearest").to_series()[0])
print(f"threshold: {threshold}")
#convert_parquet_to_tensor(train_df, transformer=_reads_only, threshold=threshold, type="train", suffix="reads_only.", even=True)
#convert_parquet_to_tensor(test_df, transformer=_reads_only, threshold=threshold, type="test", suffix="reads_only.", even=False)
#convert_parquet_to_tensor(test_df, transformer=_reads_only, threshold=threshold, type="test", suffix="reads_only.", even=True)
#convert_parquet_to_tensor(train_df, transformer=_flatten_data, threshold=threshold, type="train", even=True)
convert_parquet_to_tensor(test_df, transformer=_flatten_data, threshold=threshold, type="test", even=False)
