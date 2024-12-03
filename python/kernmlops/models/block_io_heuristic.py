import random
from typing import Mapping

import polars as pl

file_path_prefix = "data/tensors"

train_df = pl.read_parquet("data/rainsong_test_curated/block_io/*.parquet").filter(
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


def test_heuristic(data_df: pl.DataFrame, *, threshold: float):
    raw_data = data_df.select([
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
    ])
    total_correct = {
        126: 0,
        127: 0,
        128: 0,
        250: 0,
        254: 0,
        255: 0,
        256: 0,
        300: 0,
        350: 0,
        400: 0,
        450: 0,
        500: 0,
        600: 0,
        750: 0,
        1000: 0,
        1250: 0,
        1500: 0,
        1750: 0,
        2000: 0,
    }
    segments = total_correct.keys()
    total_incorrect_slow = {
        key: 0
        for key in segments
    }
    total_incorrect_fast = {
        key: 0
        for key in segments
    }
    total_count = 0
    total_fast = 0
    total_slow = 0
    for row in raw_data.iter_rows(named=True):
        actual_block_latency = row["block_latency_us"]
        fast_io = actual_block_latency < threshold
        slow_io = not fast_io
        if fast_io and random.randint(0, 20) < 18:
            continue
        exploded_flags = _explode_flags(row["block_io_flags"])
        if exploded_flags[0] != 0:
            continue
        total_count += 1
        if fast_io:
            total_fast += 1
        if slow_io:
            total_slow += 1

        for segment_threshold in segments:
            expect_slow = int(row["queue_length_segment_ios"]) > int(segment_threshold)
            if expect_slow == slow_io:
                total_correct[segment_threshold] = total_correct[segment_threshold] + 1
            elif expect_slow:
                total_incorrect_slow[segment_threshold] = total_incorrect_slow[segment_threshold] + 1
            else:
                total_incorrect_fast[segment_threshold] = total_incorrect_fast[segment_threshold] + 1
    print(f"Fast IO: {total_fast}")
    print(f"Slow IO: {total_slow}")
    for segment_threshold, correct in total_correct.items():
        print(f"Test Error for Segment threshold {segment_threshold}: \n Accuracy: {(100*correct/total_count):>0.1f}% Falsely Slow: {(100*total_incorrect_slow[segment_threshold]/total_count):>0.1f}%, Falsely Fast: {(100*total_incorrect_fast[segment_threshold]/total_count):>0.1f}%\n")

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
        "block_latency_us": 0,
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


threshold = 750 # 350 #p90 #int(test_df.select("block_latency_us").quantile(.9, interpolation="nearest").to_series()[0])
print(f"threshold: {threshold}")
percentiles = [.50, .60, .70, .75, .80, .85, .90, .95, .99]
for percentile in percentiles:
    latency_threshold = int(test_df.select("block_latency_us").quantile(percentile, interpolation="nearest").to_series()[0])
    print(f"p{percentile}: {latency_threshold}us")
test_heuristic(train_df, threshold=threshold)
