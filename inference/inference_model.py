#!/usr/bin/env python3
import itertools
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from memory_usage_hook import MemoryUsageHook
from torch.utils.data import TensorDataset, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

ZSWAP_OPTIONS = {
    'shrinker_enabled': ['N', 'Y'],
    'exclusive_loads': ['N', 'Y'],
    'compressor': ['842', 'deflate', 'lz4', 'lz4hc', 'lzo', 'zstd'],
    'non_same_filled_pages_enabled': ['N', 'Y'],
    'zpool': ['z3fold', 'zbud', 'zsmalloc'],
    'same_filled_pages_enabled': ['N', 'Y'],
    # Commented out because not included in current feature set
    # model/featurewiz selection
    #'max_pool_percent': ['10', '20', '30', '40', '50'],
    #'accept_threshold_percent': ['80', '90', '100']
}

def get_current_zswap_features():
    zswap_conf_dir = "/sys/module/zswap/parameters/"

    # Read zswap config into a dictionary
    zswap_features = {}
    with open(zswap_conf_dir + "enabled") as f:
        if f.readlines()[0].strip() == 'N':
            print("I'm not reading the rest of the params until you enable zswap, worm")
            sys.exit(1)
        else:
            for file in Path(zswap_conf_dir).iterdir():
                if file.is_file() and file.name != 'enabled':
                    with open(file) as f:
                        zswap_features[file.name] = f.readline().strip()

    # Convert the dictionary into a binary-encoded version
    encoded = {}
    for k,v in ZSWAP_OPTIONS.items():
        for val in v:
            encoded[f"{k}_{val}"] = float(zswap_features[k] == val)

    return encoded

def set_zswap_config(encoded_conf: dict) -> int:
    zswap_conf_dir = "/sys/module/zswap/parameters/"
    decoded = {}

    print("\n[INFO] Decoding one-hot encoded zswap configuration...")
    for option, choices in ZSWAP_OPTIONS.items():
        selected = None
        for val in choices:
            key = f"{option}_{val}"
            if encoded_conf.get(key, 0.0) == 1.0:
                selected = val
                break
        if selected is None:
            print(f"[WARN] No value set to 1.0 for option '{option}', skipping config.")
            return 1
        decoded[option] = selected
        print(f"  - {option} => {selected}")

    print("\n[INFO] Writing decoded config to sysfs using sudo tee...")
    for option, val in decoded.items():
        path = str(Path(zswap_conf_dir) / option)
        print(f"  - Setting {option} to '{val}' at {path}")
        try:
            subprocess.run(
                ["sudo", "tee", path],
                input=f"{val}\n".encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to write {val} to {path}")
            return 1

    print("\n[INFO] zswap configuration applied successfully.")
    return 0

# Use a for-loop  with zswap configs, not just all combos
def generate_all_valid_encoded_zswap_configs():
    all_option_values = list(ZSWAP_OPTIONS.values())
    all_option_keys = list(ZSWAP_OPTIONS.keys())
    for combo in itertools.product(*all_option_values):
        raw_dict = dict(zip(all_option_keys, combo))
        encoded = {}
        for k, vals in ZSWAP_OPTIONS.items():
            for val in vals:
                encoded[f"{k}_{val}"] = float(raw_dict[k] == val)
        yield encoded

# Read from /proc/meminfo and get features
def get_current_meminfo_features():
    feature_stat_names = ['swap_free_bytes', 'mapped_total_bytes',
                          'buffers_bytes']
    proc_info = MemoryUsageHook()
    proc_info.load()  # Initializes memory_usage list
    proc_info.poll()  # Appends /proc/meminfo stats to memory_usage
    # Get the latest MemoryUsageDataRaw from the list and run parse to get
    # the MemoryUsageData object from it
    memory_usage_data = proc_info.memory_usage[0].parse()

    # Return the desired fields
    proc_info_features = {}
    for field,value in vars(memory_usage_data).items():
        if field in feature_stat_names:
            proc_info_features[field] = value

    return proc_info_features

def compute_proc_info_stats(proc_history: dict):
    return {
        'swap_free_bytes_mean': np.mean([d["swap_free_bytes"] for d in proc_history]),
        'mapped_total_bytes_std': np.std([d["mapped_total_bytes"] for d in proc_history]),
        'buffers_bytes_max': np.max([d["buffers_bytes"] for d in proc_history]),
    }

def main():
    print("Starting model inference...")
    # Collect zswap features (encoded) and meminfo stats
    print("\nPre-populating the continuous values...")
    q = deque(maxlen=10)
    for i in range(10):
        q.appendleft(get_current_meminfo_features())
        time.sleep(1)
    #pprint(compute_proc_info_stats(q))
    #print("\nExample of collecting current zswap config...")
    #pprint(get_current_zswap_features())
    #print()

    # Load the feature tensor's training set for normalization
    print("Loading feature tensor training set for X normalization...")
    X = torch.load("redis_memory_usage_features.tensor")     # Load feature tensor from file
    X = X.float()                                            # Ensure tensor is in float format
    torch.manual_seed(42)                                    # Set fixed seed for reproducibility
    indices = torch.randperm(len(X))                         # Generate shuffled row indices
    train_size = int(0.8 * len(X))                           # Define training set size (80%)
    train_indices = indices[:train_size]                     # Select indices for training subset
    X_train = X[train_indices]                               # Extract training features

    # Find the mean and std of each continuous feature in the training set for
    # normalization purposes
    x_means = {
        "swap_free_bytes_mean": X_train.mean(dim=0)[0].item(),
        "mapped_total_bytes_std": X_train.mean(dim=0)[1].item(),
        "buffers_bytes_max": X_train.mean(dim=0)[2].item(),
    }
    x_stds = {
        "swap_free_bytes_mean": X_train.std(dim=0)[0].item(),
        "mapped_total_bytes_std": X_train.std(dim=0)[1].item(),
        "buffers_bytes_max": X_train.std(dim=0)[2].item(),
    }

    # Calculate Y_train mean and std for prediction denormalization
    print("Loading target tensor training set for Y normalization...")
    y = torch.load("redis_memory_usage_targets.tensor")
    if len(y.shape) == 1:
        y = y.unsqueeze(1)
    y = y.float()

    # Split for training
    dataset = TensorDataset(y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Extract training targets
    y_train = torch.stack([y[i] for i in train_dataset.indices])

    # Compute mean and std
    y_mean = y_train.mean()
    y_std = y_train.std()
    #print(f"Target: mean={y_mean.item():.2f}, std={y_std.item():.2f}")

    # Load the model with the correct architecture
    model = nn.Sequential(
       nn.Linear(X.shape[1], 32),
       nn.ReLU(),
       nn.Linear(32, 16),
       nn.ReLU(),
       nn.Linear(16, 1)
    ).to(device)
    #print(model, "\n")
    checkpoint = torch.load("memory_usage_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Continually run model predictions on input vector
    print("\nEntering inference loop (Ctrl-C to exit)...")
    while True:
        # Collect and normalize meminfo stats
        print("\nCollecting meminfo stats...")
        q.appendleft(get_current_meminfo_features())
        normalized_stats = {}
        for k, v in compute_proc_info_stats(q).items():
            mean = x_means[k]
            std = x_stds[k] if x_stds[k] > 1e-5 else 1.0
            normalized_stats[k] = (v - mean) / std
            #print(f"{k} (normalized): {normalized_stats[k]}")

        # Create input vector with normalized meminfo stats and zswap features
        print("\nCollecting current zswap config:")
        curr_zswap_conf = get_current_zswap_features()
        pprint(curr_zswap_conf)
        combined_dict = {**normalized_stats, **curr_zswap_conf}
        input_values = []
        for v in combined_dict.values():
            input_values.append(v)
        vec = torch.tensor(input_values, dtype=torch.float32).to(device)
        #print("\nInput tensor:")
        #print(vec, "\n")

        # Run inference on input tensor and denormalize the output
        current_prediction = 0.0
        with torch.no_grad():
            output = model(vec.unsqueeze(0))  # add batch dimension
            prediction = output.item()
            denorm_prediction = prediction * y_std.item() + y_mean.item()
            current_prediction = denorm_prediction
            print(f"Predicted runtime w/ current config: {current_prediction:.2f}")

        # Instead of just predicting the runtime for the current zswap
        # features, exhaustively search all possible zswap configurations. If
        # there is a faster predicted zswap configuration than the current one,
        # switch to that one instead.
        print("\nRunning exhaustive search for a faster zswap configuration...")
        # If a faster prediction isn't found, just use the current zswap
        # configuration
        min_prediction = current_prediction
        min_config = curr_zswap_conf

        search_start = time.time()
        for encoded_conf in generate_all_valid_encoded_zswap_configs():
            # Build hypothetical input
            combined = {**normalized_stats, **encoded_conf}
            input_vals = list(combined.values())
            test_vec = torch.tensor(input_vals, dtype=torch.float32).to(device)

            # Predict runtime
            with torch.no_grad():
                pred = model(test_vec.unsqueeze(0)).item()
                denorm_pred = pred * y_std.item() + y_mean.item()

            # Track minimum
            if denorm_pred < min_prediction:
                min_prediction = denorm_pred
                min_config = encoded_conf

        print(f"Search time: {time.time() - search_start:.4f} sec")
        print(f"Minimum predicted runtime: {min_prediction:.2f}")
        print("Best zswap config:")
        pprint(min_config)

        # If a new best config is found, just set it
        # Convert floats in config to ints because python is stupid
        if [int(v) for v in min_config.values()] != [int(v) for v in curr_zswap_conf.values()]:
            print("Predicted a faster config!")
            failed = set_zswap_config(min_config)
            if not failed:
                print("Successfully assigned new zswap config!")
            else:
                print("Failed to set new zswap config :-(")
        else:
            print("Current config is already optimal!")


        print("===== ITER END ======\n\n")

        # Wait 10 seconds, then run inference again
        time.sleep(10)

if __name__ == "__main__":
    main()
