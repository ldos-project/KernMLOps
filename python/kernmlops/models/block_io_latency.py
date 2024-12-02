import math
from typing import Mapping

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# train_df = pl.read_parquet("data/rainsong_curated/block_io/*.parquet").filter(
#     pl.col("device").is_in([
#         271581184,
#         271581185,
#         271581186,
#     ])
# )
# print(len(train_df))

test_df = pl.read_parquet("data/rainsong_test_curated/block_io/*.parquet").filter(
    pl.col("device").is_in([
        271581184,
        271581185,
        271581186,
    ])
)
print(len(test_df))


class BlockIODataset(Dataset):
    def __init__(self, data_df: pl.DataFrame, device: str):
        self.device = device
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
        latency_data = list[float]()
        for index in range(len(raw_data) - 3):
            predict_index = index + 3
            predictor_data = raw_data[index:predict_index + 1]
            actual_block_latency = predictor_data[-1]["block_latency_us"]
            cleaned_predictor_data = self._flatten_data(predictor_data)

            feature_data.append(cleaned_predictor_data)
            latency_data.append(actual_block_latency)
        self.features = torch.tensor(feature_data, dtype=torch.float32, device=device)
        self.latencies = torch.tensor(latency_data, dtype=torch.float32, device=device)


    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def row_length(cls) -> int:
        return len(cls._flatten_data([
            cls._null_data(),
            cls._null_data(),
            cls._null_data(),
            cls._null_data(),
        ]))

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
            if row["device"] == device and row["collection_id"] == collection_id:
                _append_row(row)
            else:
                _append_row(cls._null_data())

        return data


    def __getitem__(self, index: int):
        return self.features[index], self.latencies[index]


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
        #self.flatten = nn.Flatten()
        hidden_dim =  256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(BlockIODataset.row_length(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model: NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        loss.backward()
        if math.isnan(loss.item()):
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("Exiting")
            exit(1)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5000 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        first = True
        for X, y in dataloader:
            pred = model(X)
            if first:
                print("ACTUAL")
                print(pred)
                print("EXPECTED")
                print(y)
                first = False
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct += (torch.isclose(pred, y.unsqueeze(1), atol=3, rtol=.05)).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learning_rate = 1e-4
epochs = 3
batch_size = 256

# TODO(Patrick): remove this
model = NeuralNetwork().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#train_data = BlockIODataset(train_df, device=device)
test_data = BlockIODataset(test_df, device=device)

print("Loaded dataset")


train_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(model)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
