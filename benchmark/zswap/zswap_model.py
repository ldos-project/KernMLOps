import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Load tensors
X = torch.load("memory_usage_features.tensor")
y = torch.load("memory_usage_targets.tensor")

# Train/val split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model definition
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(30):
    model.train()
    total_train_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            total_val_loss += loss_fn(model(xb), yb).item()

    print(f"Epoch {epoch+1}: Train Loss = {total_train_loss:.2f}, Val Loss = {total_val_loss:.2f}")

# Save model
torch.save(model.state_dict(), "memory_usage_model.pt")
print("Model saved.")

# Inference on training set
train_predictions = []
train_actuals = []
with torch.no_grad():
    for xb, yb in train_loader:
        train_predictions.append(model(xb))
        train_actuals.append(yb)
train_predictions = torch.cat(train_predictions).squeeze().cpu()
train_actuals = torch.cat(train_actuals).squeeze().cpu()

print("\nSample predictions on training set:")
for i in range(min(10, len(train_predictions))):
    print(f"Predicted: {train_predictions[i]:.2f} sec, Actual: {train_actuals[i]:.2f} sec")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(train_actuals, train_predictions, alpha=0.5)
plt.plot([train_actuals.min(), train_actuals.max()], [train_actuals.min(), train_actuals.max()], 'r--')
plt.xlabel("Actual Runtime (sec)")
plt.ylabel("Predicted Runtime (sec)")
plt.title("Predicted vs Actual Runtime")
plt.grid(True)
plt.tight_layout()
plt.show()
