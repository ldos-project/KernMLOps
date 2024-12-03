from typing import Mapping

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
print(len(train_df))

test_df = pl.read_parquet("data/rainsong_test_curated/block_io/*.parquet").filter(
    pl.col("device").is_in([
        271581184,
        271581185,
        271581186,
    ])
)
print(len(test_df))


def convert_parquet_to_tensor(data_df: pl.DataFrame, *, threshold: float, type: str):
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
    for index in range(len(raw_data) - 3):
        predict_index = index + 3
        predictor_data = raw_data[index:predict_index + 1]
        actual_block_latency = predictor_data[-1]["block_latency_us"]
        cleaned_predictor_data = _flatten_data(predictor_data)
        fast_io = actual_block_latency < threshold
        slow_io = not fast_io

        feature_data.append(cleaned_predictor_data)
        latency_data.append([1 if fast_io else 0, 1 if slow_io else 0])
    features = torch.tensor(feature_data, dtype=torch.float32)
    latencies = torch.tensor(latency_data, dtype=torch.float32)
    torch.save(features, f"{file_path_prefix}/rainsong_{type}_features.tensor")
    torch.save(latencies, f"{file_path_prefix}/rainsong_{type}_latencies_{threshold}.tensor")

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
        "block_latency_us": 0,
        "collection_id": 0,
    }

def _flatten_data(predictor_data: list[Mapping[str, int]]) -> list[int]:
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
        data.append(row["block_io_flags"])
        data.append(row["queue_length_segment_ios"])
        data.append(row["queue_length_4k_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < start_ts_us else 0)

    for row in predictor_data:
        if row["device"] == device and row["collection_id"] == collection_id:
            _append_row(row)
        else:
            _append_row(_null_data())

    return data


threshold = int(test_df.select("block_latency_us").quantile(.95, interpolation="nearest").to_series()[0])
print(f"threshold: {threshold}")
train_data = convert_parquet_to_tensor(train_df, threshold=threshold, type="train")
test_data = convert_parquet_to_tensor(test_df, threshold=threshold, type="test")
