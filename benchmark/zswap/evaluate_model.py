import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── Load tensors ───────────────────────────────────────────────────────────────
X   = torch.load("memory_usage_features.tensor")
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

y   = torch.load("memory_usage_targets.tensor")
y_mean, y_std = y.mean(), y.std()
y   = (y - y_mean) / y_std

cid = torch.load("memory_usage_cids.tensor")          # ← NEW

# ─── Make the same 80 / 20 experiment-level split ──────────────────────────────
g = torch.Generator().manual_seed(0)
perm = cid.unique()[torch.randperm(cid.unique().size(0), generator=g)]
cut  = int(0.8 * len(perm))
train_ids, val_ids = perm[:cut], perm[cut:]

train_mask = torch.isin(cid, train_ids)
val_mask   = ~train_mask

train_ds = TensorDataset(X[train_mask], y[train_mask])
val_ds   = TensorDataset(X[val_mask],   y[val_mask])

train_loader = DataLoader(train_ds, batch_size=64)
val_loader   = DataLoader(val_ds,   batch_size=64)

# ─── Load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
    torch.nn.Linear(256,128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
    torch.nn.Linear(128,64),  torch.nn.ReLU(),
    torch.nn.Linear(64,32),   torch.nn.ReLU(),
    torch.nn.Linear(32,16),   torch.nn.ReLU(),
    torch.nn.Linear(16,1)
).to(device)
model.load_state_dict(torch.load("memory_usage_model.pt"))
model.eval()

# ─── Inference + plots (unchanged) ─────────────────────────────────────────────
def infer_and_plot(loader, label):
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu()*y_std + y_mean)
            acts.append(yb*y_std + y_mean)
    preds, acts = torch.cat(preds).squeeze(), torch.cat(acts).squeeze()

    plt.figure(figsize=(6,6))
    plt.scatter(acts, preds, alpha=0.5)
    plt.plot([acts.min(), acts.max()], [acts.min(), acts.max()], 'r--')
    plt.xlabel("Actual"), plt.ylabel("Predicted"), plt.title(f"{label} set")
    plt.tight_layout()
    plt.savefig(f"{label.lower()}_pred.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.scatter(acts, preds-acts, alpha=0.5)
    plt.axhline(0, color='r', ls='--')
    plt.xlabel("Actual"), plt.ylabel("Residual")
    plt.title(f"{label} residuals")
    plt.tight_layout()
    plt.savefig(f"{label.lower()}_resid.png", dpi=150)
    plt.close()

infer_and_plot(train_loader, "Training")
infer_and_plot(val_loader,   "Validation")
