import argparse
import gc
import re
from pathlib import Path

import pandas as pd
import polars as pl
import torch

# Provides the option to collect features for one host rather than all
parser = argparse.ArgumentParser()
parser.add_argument("--host", help="Restrict to a specific host (e.g. pc701.emulab.net)")
args = parser.parse_args()

# Sets the directory to look for dataframes and the desired metric dataframe
data_dir = Path("curated_data/")
zswap_dir = Path("benchmark/zswap/results/")
selected_metric = "memory_usage"

# Sets the host and finds all the experiment IDs that match
# ONLY checks experiment ids for metric under REDIS workload
experiment_ids = set()
host_glob = args.host if args.host else "*"
experiment_paths = data_dir.glob(f"{host_glob}/redis/*/{selected_metric}.*.parquet")

# Parses out the experiment_id from the full path
for path in experiment_paths:
    parts = path.parts
    if len(parts) >= 5:
        exp_id = parts[-2]
        experiment_ids.add(exp_id)
print(f"Found {len(experiment_ids)} experiment IDs")
if not experiment_ids:
    print("No experiments found!")
    exit(1)

# Main loop through all experiment_ids to parse and collect tensors
all_experiments_data = []
for exp_id in sorted(experiment_ids):
    # Finds all the parquet files for the selected metric under the experiment id
    files = list(data_dir.glob(f"*/redis/{exp_id}/{selected_metric}.*.parquet"))
    if not files:
        print("  No files found, skipping experiment")
        continue

    # Reads each dataframe from parquet file matching the selected metric
    dfs = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"    Error reading {f}: {e}")
    if not dfs:
        print("  No data loaded, skipping experiment")
        continue

    try:
        # Combines all metric data for an experiment id into one dataframe
        combined = pl.concat(dfs)

        # Add engineered features (only if columns exist)
        cols = combined.columns
        feature_exprs = []

        if "anon_bytes" in cols and "mem_total_bytes" in cols:
            feature_exprs.append((pl.col("anon_bytes") / pl.col("mem_total_bytes")).alias("anon_frac"))
        if "swap_free_bytes" in cols and "swap_total_bytes" in cols:
            feature_exprs.append((pl.col("swap_free_bytes") / (pl.col("swap_total_bytes") + 1)).alias("swap_free_frac"))
        if "pgfault" in cols and "pgmajfault" in cols:
            feature_exprs.append((pl.col("pgfault") - pl.col("pgmajfault")).alias("minor_faults"))

        if feature_exprs:
            combined = combined.with_columns(*feature_exprs)   # <- unpack
        else:
            print(f"Skipping feature engineering for {exp_id}: required columns missing")
            continue

        # Now aggregate by collection_id (experiment-level features)
        # Fix 1: Replace deprecated pl.NUMERIC_DTYPES with pl.selectors.numeric()
        num = pl.col(pl.selectors.numeric())
        combined = (
            combined
            .group_by("collection_id")
            .agg([
                num.mean().suffix("_mean"),
                num.std().suffix("_std"),
                num.min().suffix("_min"),
                num.max().suffix("_max"),
                num.median().suffix("_med"),
                num.quantile(0.95).suffix("_q95"),
            ])
        )

        # Finds all process_metadata and process_trace files in experiment
        meta_files = list(data_dir.glob(f"*/redis/{exp_id}/process_metadata.*.parquet"))
        trace_files = list(data_dir.glob(f"*/redis/{exp_id}/process_trace.*.parquet"))
        if not meta_files or not trace_files:
            print(f"Missing metadata or trace files for {exp_id}, skipping runtime calc")
            continue

        # Collect dataframes for all process_metadata files in exp_id
        meta_dfs = []
        for meta_file in meta_files:
            try:
                df = pl.read_parquet(meta_file)
                meta_dfs.append(df)
            except Exception as e:
                print(f"    Error reading {meta_file.name}: {e}")

        # Concatenate all the process_metadata dataframes
        meta_df = pl.concat(meta_dfs)

        # Filter for java proc parent PIDs
        ppid_series = meta_df.filter(pl.col("name") == "java").select("parent_pid")

        # Search for redis-server processes if we don't find java processes
        if ppid_series.height == 0:
            print("No java process found, trying to find redis processes...")
            ppid_series = meta_df.filter(pl.col("name") == "redis-server").select("pid")

        # Fix 2: Better handling of parent PID retrieval
        if ppid_series.height > 0:
            # Get the column name (either "parent_pid" or "pid")
            col_name = ppid_series.columns[0]
            parent_pids = ppid_series[col_name].unique().to_list()
        else:
            parent_pids = []
            print(f"No processes found for {exp_id}")

        # Now collect all dataframes from process_trace parquets in exp_id
        trace_dfs = []
        for trace_file in trace_files:
            try:
                df = pl.read_parquet(trace_file)
                # Only appends dataframes that match our expected shape
                if df.shape[1] >= 6 and df.height > 0:
                    trace_dfs.append(df)
            except Exception as e:
                print(f"    Error reading {trace_file.name}: {e}")
        if not trace_dfs:
            print("  No trace data loaded, skipping runtime calc")
            continue

        # Concatenate all the trace dataframes
        trace_df = pl.concat(trace_dfs)

        # Now find the parent PID in process_trace and find the start and end to
        # calculate the benchmark runtime
        runtimes = []
        runtime_s = None
        for pid in parent_pids:
            # Filter by matching PID in process_trace dataframe
            pid_trace = trace_df.filter(pl.col("pid") == pid)
            # Find the start and end of the matching PID
            start_df = pid_trace.filter(pl.col("cap_type") == "start").select("ts_ns")
            end_df = pid_trace.filter(pl.col("cap_type") == "end").select("ts_ns")
            # If there are both only 1, then calculate and save the runtime
            if start_df.height == 1 and end_df.height == 1:
                start = start_df[0, 0]
                end = end_df[0, 0]
                runtime_s = (end - start) / 1e9
                runtimes.append((pid, runtime_s))
        if runtimes:
            # If we found runtimes, then just take the max and save it
            runtime_s = max(rt for _, rt in runtimes)
            combined = combined.with_columns(pl.lit(runtime_s).alias("runtime_seconds"))
            all_experiments_data.append(combined)
        else:
            print(f"{exp_id}: No valid runtimes found")

        # Sweeping
        del combined
        for df in dfs:
            del df
        dfs.clear()
        gc.collect()

    except Exception as e:
        print(f"  Error processing experiment {exp_id}: {e}")

# Check data collected for all experiments
if not all_experiments_data:
    print("No data processed!")
    exit(1)
else:
    print(f"Found data on {len(all_experiments_data)} experiments")

# Concatenate all experiment data into one dataframe
try:
    all_data = pl.concat(all_experiments_data)
    # print("\nColumns in the dataset:")
    # for col_name, dtype in all_data.schema.items():
    #     print(f"  {col_name}: {dtype}")

    # Convert ALL the data into a pandas dataframe for featurewiz compatibility
    print("Generating pandas dataframe of all data...")
    pd_df = all_data.to_pandas()

    # Now we want to add the associated zswap config with the experiment and its
    # memory usage stats. To do this we will match collection IDs of experiments
    # with those in logfiles

    # Pattern of the collection ID we want to match on in the file
    uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
    zswap_rows = []
    # Loop through all log files in the zswap file directory
    for file in zswap_dir.iterdir():
        # ONLY check zswap files that START with redis
        if file.is_file() and file.name.startswith("redis") and file.name.endswith(".txt"):
            # Split configs into different knobs
            parts = file.stem.split("_")[1:-1]
            if len(parts) < 8:
                print(f"[SKIPPED] {file.name}: unexpected filename format")
                continue

            try:
                with open(file, "r") as f:
                    lines = f.readlines()
                    # We search for the UUID line starting from the bottom of
                    # the file since its not always on the same line
                    uuid_line = next((line.strip() for line in reversed(lines) if uuid_pattern.match(line.strip())), None)
                    if not uuid_line:
                        raise ValueError("No valid UUID found in file.")
                    # Then we save each zswap knob and the related UUID!
                    zswap_rows.append({
                        "collection_id": uuid_line,
                        "compressor": parts[0],
                        "zpool": parts[1],
                        "max_pool_percent": parts[2],
                        "accept_threshold_percent": parts[3],
                        "shrinker_enabled": parts[4],
                        "exclusive_loads": parts[5],
                        "same_filled_pages_enabled": parts[6],
                        "non_same_filled_pages_enabled": parts[7]
                    })
            except Exception:
                # print(f"[ERROR] {file.name}: {e}")
                continue
    # After collecting all the zswap rows, turn them into a dataframe
    zswap_df = pd.DataFrame(zswap_rows)
    # print(f"Loaded {len(zswap_df)} zswap config entries with individual knobs.")

    # Next we're going to run featurewiz on the combined zswap_df and pd_df
    # Basically what this should do is combine the zswap knobs we found for a
    # certain collection_id with the metric stats we found for that same
    # collection_id
    import featurewiz
    print("\nRunning FeatureWiz...")
    target_var = "runtime_seconds"
    exclude_columns = [ "collection_id" ]
    overlap = set(pd_df["collection_id"]).intersection(set(zswap_df["collection_id"]))
    pd_df = pd_df.merge(zswap_df, on="collection_id", how="inner")
    print("\nColumns in the dataset:")
    features_df = pd_df.drop(columns=[col for col in exclude_columns if col in pd_df.columns])

    # Call featurewiz with all memory_usage + zswap features and targeting
    # runtime_seconds
    outputs = featurewiz.featurewiz(
        dataname=features_df,
        target=target_var,
        corr_limit=0.7,
        verbose=0
    )

    # Print the selected features
    selected_features = outputs[0]
    print(f"\nSelected {len(selected_features)} features for predicting {target_var}:")
    print(selected_features)
    selected_df = pd_df[selected_features + [target_var, "collection_id"]]
    selected_output = f"{selected_metric}_selected_features_all_experiments.csv"
    selected_df.to_csv(selected_output, index=False)
    print(f"Saved selected features dataset to {selected_output}")

    # Now we save the tensors
    # We first one-hot encode all the non-numeric columns
    encoded_df = pd.get_dummies(
        selected_df.drop(columns=["collection_id", target_var]),  # drop target too
        dtype="float32"                                           # param name is dtype
    )
    print(encoded_df.columns.tolist())


    cid_codes = (
        selected_df["collection_id"]
        .astype("category")
        .cat.codes
        .astype("int32")
    )
    cid_tensor = torch.tensor(cid_codes.values, dtype=torch.int32)
    torch.save(cid_tensor, f"{selected_metric}_cids.tensor")
    print(f"Saved CIDs to {selected_metric}_cids.tensor")

    # Then convert to tensors
    features_tensor = torch.tensor(
        encoded_df.values, dtype=torch.float32
    )
    targets_tensor = torch.tensor(
        selected_df[target_var].values, dtype=torch.float32
    ).unsqueeze(1)
    torch.save(features_tensor, f"{selected_metric}_features.tensor")
    torch.save(targets_tensor, f"{selected_metric}_targets.tensor")
    print(f"Saved features to {selected_metric}_features.tensor")
    print(f"Saved targets to {selected_metric}_targets.tensor")
except Exception as e:
    print(f"Error combining experiments: {e}")

print("\nDone!")
