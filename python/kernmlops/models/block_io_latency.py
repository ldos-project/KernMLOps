import math
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class BlockIODataset(Dataset):
    def __init__(self, *, features: torch.Tensor, latencies: torch.Tensor, device: str):
        self.features = features.to(device)
        self.latencies = latencies.to(device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.latencies[index]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        hidden_dim = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*18, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RReLU(),
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
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        if math.isnan(loss.item()):
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("Exiting")
            exit(1)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, test_content):
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
            pred: torch.Tensor = model(X)
            if first:
                print("ACTUAL")
                print(pred)
                print("EXPECTED")
                print(y)
                first = False
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y[:,1]).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error ({test_content}): \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learning_rate = 1e-5
epochs = 10
batch_size = 4096

model = NeuralNetwork().to(device)
pos_weights = torch.tensor([1, 19], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

file_path_prefix = "/var/local/pkenney/tensors"
train_feature_tensor = torch.load(f"{file_path_prefix}/rainsong_train_features.flags.reduced_reads.tensor")
train_latency_tensor = torch.load(f"{file_path_prefix}/rainsong_train_latencies_350.flags.reduced_reads.tensor")
test_feature_tensor = torch.load(f"{file_path_prefix}/rainsong_test_features.flags.tensor")
test_latency_tensor = torch.load(f"{file_path_prefix}/rainsong_test_latencies_350.flags.tensor")
#test_feature_reads_tensor = torch.load(f"{file_path_prefix}/rainsong_test_features.flags.reads_only.tensor")
#test_latency_reads_tensor = torch.load(f"{file_path_prefix}/rainsong_test_latencies_350.flags.reads_only.tensor")
test_feature_even_reads_tensor = torch.load(f"{file_path_prefix}/rainsong_test_features.flags.even.reads_only.tensor")
test_latency_even_reads_tensor = torch.load(f"{file_path_prefix}/rainsong_test_latencies_350.flags.even.reads_only.tensor")

train_data = BlockIODataset(features=train_feature_tensor, latencies=train_latency_tensor, device=device)
test_data = BlockIODataset(features=test_feature_tensor, latencies=test_latency_tensor, device=device)
#test_read_data = BlockIODataset(features=test_feature_reads_tensor, latencies=test_latency_reads_tensor, device=device)
test_even_read_data = BlockIODataset(features=test_feature_even_reads_tensor, latencies=test_latency_even_reads_tensor, device=device)

print("Loaded dataset")

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#test_read_dataloader = DataLoader(test_read_data, batch_size=batch_size, shuffle=True)
test_even_read_dataloader = DataLoader(test_even_read_data, batch_size=batch_size, shuffle=True)
print(model)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, "all")
    #test_loop(test_read_dataloader, model, loss_fn, "reads only")
    test_loop(test_even_read_dataloader, model, loss_fn, "even reads only")
print("Done!")
model_file_path = f"{file_path_prefix}/models/rainsong_block_model_{epochs}.{uuid.uuid4()}.model"
print(f"Writing model to {model_file_path}")
torch.save(model, model_file_path)
