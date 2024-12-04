import random
from pathlib import Path
from typing import Any, Mapping, Protocol

import polars as pl
import torch

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
        return "segment_spartan_flags"

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
        return SegmentSpartanTransformer()

    @classmethod
    def row_filters(cls) -> list[RowFilter]:
        return [NoopFilter(), EvenFastReadFilter(), ReadsOnlyFilter(), EvenReadsOnlyFilter()]

    @classmethod
    def row_prediction_transformers(cls) -> list[RowPrediction]:
        return [P85Prediction(), P90Prediction(), P95Prediction()]

    @classmethod
    def num_rows(cls) -> int:
        return 4

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

BlockIOTransformer().convert_and_save_parquet(train_df, tensor_type="train", tensor_dir=tensor_dir)
BlockIOTransformer().convert_and_save_parquet(test_df, tensor_type="test", tensor_dir=tensor_dir)
