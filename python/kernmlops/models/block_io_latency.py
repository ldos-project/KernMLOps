from typing import Mapping

import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


train_df = pl.read_parquet("data/rainsong_curated/block_io/*.parquet").filter(
    pl.col("device").is_in([
        271581184,
        271581185,
        271581186,
    ])
)
print(len(train_df))


class BlockIODataset(Dataset):
    def __init__(self, data_df: pl.DataFrame):
        self.data = data_df.sort(["device", "ts_uptime_us"]).select([
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

    def __len__(self) -> int:
        return len(self.data) - 3

    @classmethod
    def _null_data(cls) -> Mapping[str, int]:
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

    @classmethod
    def _flatten_data(cls, predictor_data: list[Mapping[str, int]]) -> list[int]:
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
            print(row)
            if row["device"] == device and row["collection_id"] == collection_id:
                _append_row(row)
            else:
                _append_row(cls._null_data())

        return data

    @classmethod
    def _clean_data(cls, predictor_data: list[Mapping[str, int]]) -> list[Mapping[str, int]]:
        start_ts_us = predictor_data[-1]["ts_uptime_us"]
        device = predictor_data[-1]["device"]
        collection_id = predictor_data[-1]["collection_id"]
        cleaned_predictor_data = [
            dict(row) if row["device"] == device and row["collection_id"] == collection_id else cls._null_data
            for row in predictor_data
        ]
        for row in cleaned_predictor_data:
            del row["collection_id"]
            if row["ts_uptime_us"] + row["block_latency_us"] > start_ts_us:
                row["block_latency_us"] = 0
            row["time_since_io_us"] = -(row["ts_uptime_us"] - start_ts_us) if row["ts_uptime_us"] > 0 else 0
            del row["ts_uptime_us"]
        return cleaned_predictor_data


    def __getitem__(self, index: int):
        predict_index = index + 3
        predictor_data = self.data[index:predict_index + 1]
        actual_block_latency = predictor_data[-1]["block_latency_us"]
        cleaned_predictor_data = self._flatten_data(predictor_data)

        X = torch.tensor(cleaned_predictor_data, dtype=torch.long)
        return X, actual_block_latency


dataset = BlockIODataset(train_df)
print("Loaded dataset")
