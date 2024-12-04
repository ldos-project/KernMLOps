from collections import defaultdict

import plotille
import polars as pl

import python.kernmlops.data_import as di

curated_data = "data/curated"
r = di.read_parquet_dir(curated_data)
metrics = r.keys()

def update_metrics_dict(merged_metrics: dict[int, list[float]], df: pl.DataFrame) -> None:
    for row_idx, (timestamp, value) in enumerate(df.iter_rows()):
        if row_idx == 0:
            continue
        merged_metrics[timestamp].append(value)

def to_time_series(df: pl.DataFrame, target_metric: str, uptime_metric: str) -> pl.DataFrame:
    # every 100ms
    df = df.with_columns([pl.col(uptime_metric) // 1e5])
    df = df.group_by(uptime_metric).agg(pl.col(target_metric).max())
    df = df.sort(uptime_metric)
    df = df.with_columns([pl.col(target_metric).diff()])

    merged_metrics = defaultdict(list)
    update_metrics_dict(merged_metrics, df)
    return merged_metrics

def merge_metrics_by_cpu(
        df: pl.DataFrame, target_metric: str, uptime_metric: str = "ts_uptime_us"
    ) -> dict[int, float]:
    metrics_by_cpu = defaultdict(lambda: defaultdict(int))
    for cpu_id, llc_miss_per_cpu in df.group_by("cpu"):
        merged_metrics = to_time_series(llc_miss_per_cpu, target_metric, uptime_metric)
        metrics_by_cpu[cpu_id[0]] = merged_metrics

    merged_metrics = defaultdict(list)
    for cpu_id, timestamps in metrics_by_cpu.items():
        for timestamp, value in timestamps.items():
            merged_metrics[timestamp].extend(value)
    return dict(sorted(merged_metrics.items()))

def plot_metrics(merged_metrics: dict[int, list[float]], target_metric: str) -> None:
    min_timestamp = min(merged_metrics.keys())
    merged_xvals = []
    merged_yvals = []

    for timestamp, values in merged_metrics.items():
        merged_xvals.append(timestamp - min_timestamp)
        merged_yvals.append(sum(values) / len(values))

    print(
        plotille.plot(
            merged_xvals,
            merged_yvals,
            height=15,
            width=200,
            interp="linear",
            lc="cyan",
            x_min=min(merged_xvals),
            x_max=max(merged_xvals),
            y_min=min(merged_yvals),
            y_max=max(merged_yvals),
            X_label="timestamp",
            Y_label=target_metric,
        )
    )

perf_metric_pairs = [
    ("llc_misses", "cumulative_llc_misses"),
    ("l1i_misses", "cumulative_l1i_misses"),
    ("l1d_misses", "cumulative_l1d_misses"),
    ("branch_misses", "cumulative_branch_misses"),
    ("local_mem_misses", "cumulative_local_mem_misses"),
]
disk_metric_pairs = [
    ("disk_usage", "num_sector_read"),
    ("disk_usage", "num_sector_write"),
    ("memory_usage", "mem_free_bytes"),
]

for table_name, target_metric in perf_metric_pairs:
    merged_metrics = merge_metrics_by_cpu(r[table_name], target_metric)
    plot_metrics(merged_metrics, target_metric)

for table_name, target_metric in disk_metric_pairs:
    merged_metrics = to_time_series(r[table_name], target_metric, "ts_uptime_us")
    plot_metrics(merged_metrics, target_metric)
