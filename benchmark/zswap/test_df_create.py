import re
from pathlib import Path

import pandas as pd

uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
config_dir = Path("results/")

rows = []

for file in config_dir.iterdir():
    if file.is_file() and file.name.startswith("redis") and file.name.endswith(".txt"):
        parts = file.stem.split("_")
        if len(parts) < 8:
            print(f"[SKIPPED] {file.name}: unexpected filename format")
            continue
        try:
            with open(file, "r") as f:
                lines = f.readlines()
                uuid_line = next((line.strip() for line in reversed(lines) if uuid_pattern.match(line.strip())), None)
                if not uuid_line:
                    raise ValueError("No valid UUID found in file.")

                # Extract knobs
                compressor     = parts[1]
                zpool          = parts[2]
                max_pool       = parts[3]
                accept_thresh  = parts[4]
                shrinker       = parts[5]
                exclusive      = parts[6]
                same_filled    = parts[7]

                rows.append({
                    "collection_id": uuid_line,
                    "compressor": compressor,
                    "zpool": zpool,
                    "max_pool_percent": max_pool,
                    "accept_threshold_percent": accept_thresh,
                    "shrinker_enabled": shrinker,
                    "exclusive_loads": exclusive,
                    "same_filled_pages_enabled": same_filled
                })
        except Exception as e:
            print(f"[ERROR] {file.name}: {e}")
            continue

# Create and save DataFrame
df = pd.DataFrame(rows)
df.to_csv("zswap_configs_split.csv", index=False)
print(f"Saved {len(df)} rows to zswap_configs_split.csv")
