import random
from pathlib import Path
from typing import Any, Mapping, Protocol

import polars as pl
import torch

import argparse
from enum import Enum

# Setup operation
class Mode(Enum):
    WHIZ = "whiz"
    MINT = "mint"
    INVE = "investigate"

def mode_type(value: str) -> Mode:
    try:
        return Mode(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid mode: {value}. Choose from {[m.value for m in Mode]}")

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=mode_type, default=Mode.WHIZ, help=f"Select mode. Options: {[m.value for m in Mode]}")

args = parser.parse_args()


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

def convert_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["device", "ts_uptime_us"]).with_columns([
            (pl.col("block_io_flags") & REQ_OP_MASK).replace_strict(req_opf).alias("op"),
            pl.when((pl.col("block_io_flags") & REQ_SYNC > 0)).then(True).otherwise(False).alias("sync_flag"),
            pl.when((pl.col("block_io_flags") & REQ_META > 0)).then(True).otherwise(False).alias("metadata_flag"),
            pl.when((pl.col("block_io_flags") & REQ_FUA > 0)).then(True).otherwise(False).alias("fua_flag"),
            pl.when((pl.col("block_io_flags") & REQ_PRIO > 0)).then(True).otherwise(False).alias("priority_flag"),
            pl.when((pl.col("block_io_flags") & REQ_NOMERGE > 0)).then(True).otherwise(False).alias("nomerge_flag"),
            pl.when((pl.col("block_io_flags") & REQ_IDLE > 0)).then(True).otherwise(False).alias("idle_flag"),
            pl.when((pl.col("block_io_flags") & REQ_RAHEAD > 0)).then(True).otherwise(False).alias("readahead_flag"),
            pl.when((pl.col("block_io_flags") & REQ_BACKGROUND > 0)).then(True).otherwise(False).alias("background_flag"),
            pl.when((pl.col("block_io_flags") & REQ_NOWAIT > 0)).then(True).otherwise(False).alias("nowait_flag"),
            pl.col("sector").shift(1, fill_value=0).alias("sector_offset"),
            pl.col("ts_uptime_us").shift(1, fill_value=0).alias("ts_offset"),
        ]).with_columns([
            (pl.col("ts_uptime_us") - pl.col("ts_offset")).alias("ts_uptime_us"),
            (pl.col("sector") - pl.col("sector_offset")).alias("sector_offset"),
        ]).select([
            "cpu",
            "device",
            "sector",
            "sector_offset",
            "segments",
            "block_io_bytes",
            "ts_uptime_us",
            "queue_length_segment_ios",
            "queue_length_4k_ios",
            "block_latency_us",
            "op",
            "sync_flag",
            "metadata_flag",
            "fua_flag",
            "priority_flag",
            "nomerge_flag",
            "idle_flag",
            "readahead_flag",
            "background_flag",
            "nowait_flag",
        ])
    df = df.filter(pl.col("op") == "Read")
    for i in ["sync_flag", "metadata_flag", "fua_flag", "priority_flag", "nomerge_flag",
              "idle_flag", "readahead_flag", "background_flag", "nowait_flag"]:
        new_df = df.filter(pl.col(i))
        if new_df.height == 0:
            df = df.drop(i)

    concat_df = pl.DataFrame()
    for i in [271581184, 271581185, 271581186]:
        out_df = df.filter(pl.col("device") == i)
        for j in out_df.columns:
            for k in range(1, 8):
                out_df = out_df.with_columns(pl.col(j).shift(i).alias(f"{j}_lag_{k}"))

        concat_df = pl.concat([concat_df, out_df])
    return concat_df

class RowTransformer(Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def feature_length(cls) -> int: ...

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]: ...

class RowFilter(Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def skip_row(cls, row: Mapping[str, Any]) -> bool: ...


class RowPrediction(Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def prediction(cls, row: Mapping[str, Any]) -> list[float]: ...


class DatasetTransformer(Protocol):

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def row_transformer(cls) -> RowTransformer: ...

    @classmethod
    def row_filters(cls) -> list[RowFilter]: ...

    @classmethod
    def row_prediction_transformers(cls) -> list[RowPrediction]: ...

    @classmethod
    def num_rows(cls) -> int: ...

    def convert_and_save_parquet(self, data_df: pl.DataFrame, tensor_dir: str) -> str: ...


class SegmentSpartanTransformer(RowTransformer):

    @classmethod
    def name(cls) -> str:
        return "segment_spartan"

    @classmethod
    def feature_length(cls) -> int:
        return 3

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]:
        data = list[float]()
        data.append(row["segments"])
        data.append(row["queue_length_segment_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < present_ts_us else 0)
        return data

class SegmentMinimalFlagsTransformer(RowTransformer):

    @classmethod
    def name(cls) -> str:
        return "segment_minimal_flags"

    @classmethod
    def feature_length(cls) -> int:
        return 6

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]:
        data = list[float]()
        data.append(row["device"])
        data.append(row["segments"])
        data.append(-(row["ts_uptime_us"] - present_ts_us) if row["ts_uptime_us"] > 0 else 0)
        data.append(row["block_io_flags"] & REQ_OP_MASK)
        data.append(row["queue_length_segment_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < present_ts_us else 0)
        return data


class SegmentSpartanFlagsTransformer(RowTransformer):

    @classmethod
    def name(cls) -> str:
        return "segment_spartan_flags"

    @classmethod
    def feature_length(cls) -> int:
        return 6

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]:
        data = list[float]()
        data.append(row["segments"])
        data.append(row["block_io_flags"] & REQ_OP_MASK)
        data.append(row["queue_length_segment_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < present_ts_us else 0)
        return data


class SegmentSpartanAllFlagsTransformer(RowTransformer):

    @classmethod
    def name(cls) -> str:
        return "segment_spartan_all_flags"

    @classmethod
    def feature_length(cls) -> int:
        return 12

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]:
        data = list[float]()
        data.append(row["segments"])
        data.extend(_explode_flags(row["block_io_flags"]))
        data.append(row["queue_length_segment_ios"])
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < present_ts_us else 0)
        return data

class SegmentSpartanFwhizTransformer(RowTransformer):

    @classmethod
    def name(cls) -> str:
        return "segment_spartan_fwhiz"

    @classmethod
    def feature_length(cls) -> int:
        return 5

    @classmethod
    def convert_row(cls, row: Mapping[str, Any], present_ts_us: int) -> list[float]:
        data = list[float]()
        data.append(row["queue_length_segment_ios"])
        data.append(row["queue_length_4k_ios"])
        data.append(1 if row["block_io_flags"] & REQ_SYNC > 0 else 0)
        data.append(1 if row["block_io_flags"] & REQ_NOMERGE > 0 else 0)
        data.append(row["block_latency_us"] if row["ts_uptime_us"] + row["block_latency_us"] < present_ts_us else 0)
        return data

class P95Prediction(RowPrediction):

    @classmethod
    def threshold(cls) -> int:
        return 1460

    @classmethod
    def name(cls) -> str:
        return f"p95_{cls.threshold()}us"

    @classmethod
    def prediction(cls, row: Mapping[str, Any]) -> list[float]:
        actual_block_latency = row["block_latency_us"]
        fast_io = actual_block_latency < cls.threshold()
        slow_io = not fast_io
        return [1 if fast_io else 0, 1 if slow_io else 0]


class P90Prediction(RowPrediction):

    @classmethod
    def threshold(cls) -> int:
        return 320

    @classmethod
    def name(cls) -> str:
        return f"p90_{cls.threshold()}us"

    @classmethod
    def prediction(cls, row: Mapping[str, Any]) -> list[float]:
        actual_block_latency = row["block_latency_us"]
        fast_io = actual_block_latency < cls.threshold()
        slow_io = not fast_io
        return [1 if fast_io else 0, 1 if slow_io else 0]


class P85Prediction(RowPrediction):

    @classmethod
    def threshold(cls) -> int:
        return 160

    @classmethod
    def name(cls) -> str:
        return f"p85_{cls.threshold()}us"

    @classmethod
    def prediction(cls, row: Mapping[str, Any]) -> list[float]:
        actual_block_latency = row["block_latency_us"]
        fast_io = actual_block_latency < cls.threshold()
        slow_io = not fast_io
        return [1 if fast_io else 0, 1 if slow_io else 0]


class LatencyPrediction(RowPrediction):

    @classmethod
    def name(cls) -> str:
        return "latency_us"

    @classmethod
    def prediction(cls, row: Mapping[str, Any]) -> list[float]:
        return [row["block_latency_us"], row["block_io_latency_us"]]


class NoopFilter(RowFilter):

    @classmethod
    def name(cls) -> str:
        return "all"

    @classmethod
    def skip_row(cls, row: Mapping[str, Any]) -> bool:
        return False


class EvenFastReadFilter(RowFilter):

    @classmethod
    def name(cls) -> str:
        return "even"

    @classmethod
    def skip_row(cls, row: Mapping[str, Any]) -> bool:
        if row["block_latency_us"] < 320:
            return random.randint(0, 10) < 9 # 90% chance to skip
        return False


class ReadsOnlyFilter(RowFilter):

    @classmethod
    def name(cls) -> str:
        return "reads_only"

    @classmethod
    def skip_row(cls, row: Mapping[str, Any]) -> bool:
        return (row["block_io_flags"] & REQ_OP_MASK) != 0


class EvenReadsOnlyFilter(RowFilter):

    @classmethod
    def name(cls) -> str:
        return "even_reads_only"

    @classmethod
    def skip_row(cls, row: Mapping[str, Any]) -> bool:
        if (row["block_io_flags"] & REQ_OP_MASK) != 0:
            return True
        if row["block_latency_us"] < 320:
            return random.randint(0, 10) < 9 # 90% chance to skip
        return False


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
        "block_latency_us": 10_000,
        "block_io_latency_us": 10_000,
        "collection_id": 0,
    }


def _check_already_generated_files(files: list[Path]) -> bool:
    for file in files:
        if not file.exists():
            return False
    return True


class BlockIOTransformer(DatasetTransformer):

    @classmethod
    def name(cls) -> str:
        return "block_io"

    @classmethod
    def row_transformer(cls) -> RowTransformer:
        return SegmentSpartanFwhizTransformer()

    @classmethod
    def row_filters(cls) -> list[RowFilter]:
        return [NoopFilter(), EvenFastReadFilter(), ReadsOnlyFilter(), EvenReadsOnlyFilter()]

    @classmethod
    def row_prediction_transformers(cls) -> list[RowPrediction]:
        return [P85Prediction(), P90Prediction(), P95Prediction(), LatencyPrediction()]

    @classmethod
    def num_rows(cls) -> int:
        return 1

    def convert_and_save_parquet(self, data_df: pl.DataFrame, *, tensor_type: str, tensor_dir: str) -> str:
        root_out_dir = Path(tensor_dir)
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
            "block_io_latency_us",
            "block_latency_us",
            "collection_id",
        ]).rows(named=True)

        for row_filter in self.row_filters():
            row_transformer_dir = f"{self.row_transformer().feature_length()}_{self.num_rows()}_{self.row_transformer().name()}"
            out_dir = root_out_dir / self.name() / row_transformer_dir / row_filter.name()
            out_dir.mkdir(parents=True, exist_ok=True)
            features_out_file = out_dir / f"{tensor_type}_features.tensor"
            predictions_out_files = [
                out_dir / f"{tensor_type}_predictions_{row_prediction.name()}.tensor"
                for row_prediction in self.row_prediction_transformers()
            ]
            if _check_already_generated_files(
                [features_out_file] + predictions_out_files,
            ):
                print(f"{str(features_out_file)} already generated, skipping...")
                continue


            feature_data = list[list[float]]()
            latency_data = {
                row_prediction.name(): list[list[float]]()
                for row_prediction in self.row_prediction_transformers()
            }
            for index in range(len(raw_data) - self.num_rows() + 1):
                predict_index = index + self.num_rows() - 1
                predictor_data = raw_data[index:predict_index + 1]
                if row_filter.skip_row(predictor_data[-1]):
                    continue

                cleaned_data = list[int]()
                start_ts_us = predictor_data[-1]["ts_uptime_us"]
                device = predictor_data[-1]["device"]
                collection_id = predictor_data[-1]["collection_id"]

                for row in predictor_data:
                    if row["device"] == device and row["collection_id"] == collection_id:
                        cleaned_data.extend(self.row_transformer().convert_row(row, present_ts_us=start_ts_us))
                    else:
                        cleaned_data.extend(self.row_transformer().convert_row(_null_data(), present_ts_us=start_ts_us))

                feature_data.append(cleaned_data)
                for row_prediction in self.row_prediction_transformers():
                    latency_data[row_prediction.name()].append(
                        row_prediction.prediction(predictor_data[-1])
                    )
            features = torch.tensor(feature_data, dtype=torch.float32)
            latencies = {
                latency_name: torch.tensor(latency_datum, dtype=torch.float32)
                for latency_name, latency_datum in latency_data.items()
            }
            torch.save(features, features_out_file)
            for prediction_name, latencies in latencies.items():
                torch.save(latencies, out_dir / f"{tensor_type}_predictions_{prediction_name}.tensor")
            print(f"Generated: {str(features_out_file)}")


tensor_dir = "data/tensors"

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

if args.mode is Mode.WHIZ:
    import featurewiz
    train_df_blah = convert_df(train_df)
    test_df_blah = convert_df(test_df)
    train_df_blah = train_df_blah.filter(pl.col("op") == "Read")
    raw_data = train_df_blah.to_pandas()
    outputs = featurewiz.featurewiz(dataname=raw_data, target="block_latency_us",
                                    corr_limit=1, verbose=2, test_data="", feature_engg="", category_encoders="")

    outputs[0].append("block_latency_us")
    to_select = outputs[0]

    test_df = test_df_blah.select(to_select)
    train_df = train_df_blah.select(to_select)
    test_df.write_parquet(tensor_dir / Path("test.fwhiz.parquet"))
    train_df.write_parquet(tensor_dir / Path("train.fwhiz.parquet"))

if args.mode is Mode.MINT:
    BlockIOTransformer().convert_and_save_parquet(train_df, tensor_type="train", tensor_dir=tensor_dir)
    BlockIOTransformer().convert_and_save_parquet(test_df, tensor_type="test", tensor_dir=tensor_dir)

if args.mode is Mode.INVE:
    train_df = convert_df(train_df)
    print(train_df.filter((pl.col("op") == "Read") & (pl.col("idle_flag"))))
    for i in ["sync_flag", "metadata_flag", "fua_flag", "priority_flag", "nomerge_flag",
              "idle_flag", "readahead_flag", "background_flag", "nowait_flag"]:
        new_df = train_df.filter(pl.col(i))
        if new_df.height != 0:
            print(f"{i}: {new_df.height}")
            print(new_df.select(pl.col("op").unique()))
