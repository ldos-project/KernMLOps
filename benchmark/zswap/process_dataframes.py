import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import ultraimport

di = ultraimport('__dir__/../../python/kernmlops/data_import/__init__.py')

BASE_DIR = 'data/curated/'
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

COLORS = {
    "main": "#42A5F5",
    "light": "#90CAF9",
    "dark": "#1976D2",
    "accent": "#64B5F6"
}

experiment_names = {
    "zswap_off": "Zswap OFF",
    "zswap_off_mem": "Zswap OFF + swapping",
    "zswap_on": "Zswap ON (default)",
    "zswap_accept_thresh": "% Accept Threshold",
    "zswap_max_pool_pct": "Max Pool Size",
    "zswap_compressor": "Compressor Type",
    "zswap_zpool": "Zpool Type",
    "zswap_exclusive_loads_on": "Invalidate Loads OFF",
    "zswap_non_same_filled_pages_off": "Save Same-value Pages ONLY",
    "zswap_same_filled_pages_off": "Save Same-value Pages OFF",
    "zswap_shrinker_off": "Shrinker OFF",
    "zswap_mem_tuning": "Tune Memory Size"
}

param_display_names = {
    # accept_thresh
    "thresh_40": "40% Threshold", "thresh_60": "60% Threshold",
    "thresh_70": "70% Threshold", "thresh_80": "80% Threshold",
    # max_pool_pct
    "pool_10": "10% Pool", "pool_30": "30% Pool", "pool_40": "40% Pool",
    "pool_50": "50% Pool", "pool_75": "75% Pool",
    # compressor
    "842": "842 Compressor", "deflate": "Deflate Compressor",
    "lz4": "LZ4 Compressor", "lz4hc": "LZ4HC Compressor",
    "zstd": "ZSTD Compressor",
    # zpool
    "z3fold": "Z3fold Pool", "zsmalloc": "ZSmalloc Pool",
    # memory tuning configs
    "mem_4GB_swap_8GB": "4GB RAM, 8GB Swap",
    "mem_6GB_swap_8GB": "6GB RAM, 8GB Swap",
    "mem_8GB_swap_8GB": "8GB RAM, 8GB Swap"
}

# Parameter extraction patterns
param_patterns = {
    "zswap_accept_thresh": [
        ("thresh_40", r"thresh_40"), ("thresh_60", r"thresh_60"),
        ("thresh_70", r"thresh_70"), ("thresh_80", r"thresh_80")
    ],
    "zswap_max_pool_pct": [
        ("pool_10", r"pool_10"), ("pool_30", r"pool_30"),
        ("pool_40", r"pool_40"), ("pool_50", r"pool_50"),
        ("pool_75", r"pool_75")
    ],
    "zswap_compressor": [
        ("842", r"842"), ("deflate", r"deflate"),
        ("lz4hc", r"lz4hc"), ("lz4", r"lz4(?!hc)"),
        ("zstd", r"zstd")
    ],
    "zswap_zpool": [
        ("z3fold", r"z3fold"), ("zsmalloc", r"zsmalloc")
    ],
    "zswap_mem_tuning": None
}

def extract_parameter_setting(exp_id, run_dir):
    if exp_id == "zswap_mem_tuning":
        # Special case for memory tuning
        match = re.search(r'zswap_conf_\d+_mem_(\d+)_swap_(\d+)', run_dir)
        if match:
            mem_gb, swap_gb = match.group(1), match.group(2)
            param_key = f"mem_{mem_gb}GB_swap_{swap_gb}GB"
            if param_key not in param_display_names:
                param_display_names[param_key] = f"{mem_gb}GB RAM, {swap_gb}GB Swap"
            return param_key
        return "unknown_mem_config"

    # For other parameter types, use the patterns
    if exp_id in param_patterns and param_patterns[exp_id]:
        for param_id, pattern in param_patterns[exp_id]:
            if re.search(pattern, run_dir):
                return param_id
        return f"unknown_{exp_id.replace('zswap_', '')}"
    return None

def configure_plot_quality(fig, ax):
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.7)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
    ax.tick_params(direction='out', length=4, width=1, colors='black')
    plt.tight_layout()
    return fig, ax

def add_bar_labels(ax):
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10
        )

def create_bar_chart(df, x_col, y_col, title, filename, color=COLORS["main"]):
    """Create and save a bar chart with consistent styling."""
    plt.figure(figsize=(12, 8))
    # Use the experiment_order column for ordering if it exists
    if 'experiment_order' in df.columns and x_col == 'experiment':
        order = df.sort_values('experiment_order')[x_col].tolist()
        ax = sns.barplot(data=df, x=x_col, y=y_col, color=color, order=order)
    else:
        ax = sns.barplot(data=df, x=x_col, y=y_col, color=color)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(title, fontsize=16)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=14)
    add_bar_labels(ax)
    fig = plt.gcf()
    configure_plot_quality(fig, ax)
    plt.savefig(f"{IMAGES_DIR}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_grouped_bar_chart(df, x_col, y_cols, names, title, filename):
    """Create and save a grouped bar chart comparing two metrics."""
    # Determine the order of x values based on experiment_order if applicable
    if 'experiment_order' in df.columns and x_col == 'experiment':
        df_sorted = df.sort_values('experiment_order')
        x_values = df_sorted[x_col].tolist()
        # Also need to sort the y values in the same order
        y_values1 = df_sorted[y_cols[0]].tolist()
        y_values2 = df_sorted[y_cols[1]].tolist()
    else:
        x_values = df[x_col].tolist()
        y_values1 = df[y_cols[0]].tolist()
        y_values2 = df[y_cols[1]].tolist()

    x = np.arange(len(x_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 9))

    # Create the bars
    rects1 = ax.bar(x - width/2, y_values1, width, label=names[0], color='lightblue')
    rects2 = ax.bar(x + width/2, y_values2, width, label=names[1], color='darkblue')

    # Add labels and title
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel("Duration (seconds)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=45, ha='right')
    ax.legend()

    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    # Apply styling and save
    configure_plot_quality(fig, ax)
    plt.savefig(f"{IMAGES_DIR}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_run_data(run_path, exp_id=None, param_setting=None):
    """Process a single run's data and return metrics."""
    try:
        run_data = di.read_parquet_dir(run_path)
        results = {"collection_time": None, "zswap_duration": 0.0}

        # Get collection time from system_info
        if 'system_info' in run_data:
            results["collection_time"] = run_data['system_info'].select(
                pl.col("collection_time_sec")).item()
        else:
            return None

        # Get zswap duration if available
        if 'zswap_runtime' in run_data:
            zswap_data = run_data['zswap_runtime']
            if "start_ts" in zswap_data.columns and "end_ts" in zswap_data.columns:
                zswap_with_duration = zswap_data.with_columns(
                    (pl.col("end_ts") - pl.col("start_ts")).alias("duration_ns")
                )
                total_duration_ns = zswap_with_duration.select(pl.sum("duration_ns")).item()
                results["zswap_duration"] = total_duration_ns / 1e9

        return results
    except Exception as e:
        print(f"Error processing {run_path}: {e}")
        return None

def analyze_experiments():
    """Main function to analyze all experiments and generate plots."""
    # Initialize data structures
    experiment_summaries = []
    param_specific_data = {exp_id: {} for exp_id in [
        "zswap_accept_thresh", "zswap_max_pool_pct",
        "zswap_compressor", "zswap_zpool", "zswap_mem_tuning"
    ]}

    # Process all experiments
    for exp_dir in sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]):
        # Skip directories that don't match with experiment_names
        if exp_dir not in experiment_names:
            print(f"\nSkipping experiment: {exp_dir} - Not in experiment_names dictionary")
            continue

        exp_path = os.path.join(BASE_DIR, exp_dir)
        exp_readable_name = experiment_names.get(exp_dir, exp_dir)
        print(f"\nProcessing experiment: {exp_dir} ({exp_readable_name})")

        # Data collection
        collection_times = []
        zswap_durations = []
        runs_processed = 0
        is_parameterized = exp_dir in param_specific_data

        # Process each run in this experiment
        for run_dir in sorted([d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]):
            run_path = os.path.join(exp_path, run_dir)
            print(f"  Processing run: {run_dir}")

            # Extract parameter setting if applicable
            param_setting = None
            if is_parameterized:
                param_setting = extract_parameter_setting(exp_dir, run_dir)
                print(f"    Parameter setting: {param_setting}")
                if param_setting and param_setting not in param_specific_data[exp_dir]:
                    param_specific_data[exp_dir][param_setting] = {
                        "collection_times": [], "zswap_durations": [], "runs_processed": 0
                    }

            # Process this run's data
            run_results = process_run_data(run_path, exp_dir, param_setting)
            if not run_results:
                continue

            # Store results for experiment summary
            collection_times.append(run_results["collection_time"])
            zswap_durations.append(run_results["zswap_duration"])
            runs_processed += 1

            # Store results for parameter-specific analysis
            if is_parameterized and param_setting:
                data = param_specific_data[exp_dir][param_setting]
                data["collection_times"].append(run_results["collection_time"])
                data["zswap_durations"].append(run_results["zswap_duration"])
                data["runs_processed"] += 1

        # Calculate experiment averages
        if runs_processed > 0:
            avg_collection_time = sum(collection_times) / len(collection_times)
            avg_zswap_duration = sum(zswap_durations) / len(zswap_durations) if zswap_durations else 0.0
            avg_zswap_percentage = (avg_zswap_duration / avg_collection_time) * 100 if avg_collection_time > 0 else 0

            # Add summary for this experiment
            experiment_summaries.append({
                "experiment_id": exp_dir,
                "experiment": exp_readable_name,
                "runs_processed": runs_processed,
                "avg_runtime_s": avg_collection_time,
                "avg_zswap_duration_s": avg_zswap_duration,
                "avg_zswap_percentage": avg_zswap_percentage
            })

            print(f"  Summary: Avg Runtime: {avg_collection_time:.2f}s, "
                  f"Avg Zswap: {avg_zswap_duration:.2f}s ({avg_zswap_percentage:.2f}%)")
        else:
            print(f"  No valid runs processed for {exp_dir}")

    return experiment_summaries, param_specific_data

def generate_plots(experiment_summaries, param_specific_data):
    """Generate all plots based on the experiment and parameter summaries."""
    if not experiment_summaries:
        print("No experiment data was processed successfully")
        return

    # Create summary dataframe
    summary_df = pd.DataFrame(experiment_summaries)
    print("\nExperiment Summary DataFrame:")
    print(summary_df)

    # Create a category type based on the order in experiment_names
    # This will ensure the plots are ordered according to the order in experiment_names
    experiment_order = list(experiment_names.keys())
    summary_df['experiment_order'] = pd.Categorical(
        summary_df['experiment_id'],
        categories=experiment_order,
        ordered=True
    )

    # Sort by this order
    summary_df = summary_df.sort_values('experiment_order')

    # Filter out specific experiments
    plot_df = summary_df[~summary_df["experiment_id"].isin(["exp_b", "exp_h"])]

    # Create overall experiment charts
    create_bar_chart(
        plot_df, "experiment", "avg_runtime_s",
        "Average Kernel Benchmark Runtime by Zswap Configuration",
        "zswap_config_runtime_comparison", COLORS["main"]
    )

    create_bar_chart(
        plot_df, "experiment", "avg_runtime_s",
        "Kernel Benchmark Runtime by Zswap Configuration (No Zswap Info)",
        "zswap_config_standalone_runtime", COLORS["main"]
    )

    create_bar_chart(
        plot_df, "experiment", "avg_zswap_duration_s",
        "Average Time Spent in Zswap Functions by Configuration",
        "zswap_config_function_duration", COLORS["dark"]
    )

    create_bar_chart(
        plot_df, "experiment", "avg_zswap_percentage",
        "Percentage of Runtime Spent in Zswap Functions by Configuration",
        "zswap_config_percentage_comparison", COLORS["accent"]
    )

    create_grouped_bar_chart(
        plot_df, "experiment", ["avg_runtime_s", "avg_zswap_duration_s"],
        ["Total Runtime", "Zswap Duration"],
        "Kernel Benchmark Runtime vs Zswap Function Duration by Configuration",
        "zswap_config_runtime_vs_function_duration"
    )

    # Create parameter-specific charts
    for exp_id, param_data in param_specific_data.items():
        if not param_data:
            continue

        # Calculate averages for each parameter
        param_summaries = []
        for param, data in param_data.items():
            if data["runs_processed"] > 0:
                avg_runtime = sum(data["collection_times"]) / len(data["collection_times"])
                avg_zswap = sum(data["zswap_durations"]) / len(data["zswap_durations"]) if data["zswap_durations"] else 0.0
                avg_percentage = (avg_zswap / avg_runtime * 100) if avg_runtime > 0 else 0

                param_summaries.append({
                    "param_id": param,
                    "param_name": param_display_names.get(param, param),
                    "avg_runtime_s": avg_runtime,
                    "avg_zswap_duration_s": avg_zswap,
                    "avg_zswap_percentage": avg_percentage,
                    "runs_processed": data["runs_processed"]
                })

        if not param_summaries:
            continue

        # Create parameter-specific dataframe
        param_df = pd.DataFrame(param_summaries)

        # Sort appropriately
        if exp_id == "zswap_mem_tuning":
            param_df['mem_size'] = param_df['param_id'].str.extract(r'mem_(\d+)GB', expand=False).astype(int)
            param_df = param_df.sort_values('mem_size')
        else:
            param_df = param_df.sort_values('param_id')

        exp_name = experiment_names.get(exp_id, exp_id)

        # Create parameter-specific charts
        create_bar_chart(
            param_df, "param_name", "avg_runtime_s",
            f"Average Runtime by {exp_name} Setting",
            f"zswap_{exp_name}_runtime_comparison", COLORS["light"]
        )

        create_bar_chart(
            param_df, "param_name", "avg_runtime_s",
            f"Kernel Benchmark Runtime by {exp_name} Setting (No Zswap Info)",
            f"zswap_{exp_name}_standalone_runtime", COLORS["main"]
        )

        create_bar_chart(
            param_df, "param_name", "avg_zswap_duration_s",
            f"Average Zswap Time by {exp_name} Setting",
            f"zswap_{exp_name}_function_duration", COLORS["dark"]
        )

        create_bar_chart(
            param_df, "param_name", "avg_zswap_percentage",
            f"Percentage of Runtime in Zswap by {exp_name} Setting",
            f"zswap_{exp_name}_percentage_comparison", COLORS["accent"]
        )

        create_grouped_bar_chart(
            param_df, "param_name", ["avg_runtime_s", "avg_zswap_duration_s"],
            ["Total Runtime", "Zswap Duration"],
            f"Runtime vs Zswap Duration by {exp_name} Setting",
            f"zswap_{exp_name}_runtime_vs_duration"
        )

if __name__ == "__main__":
    # Run the analysis pipeline
    experiment_summaries, param_specific_data = analyze_experiments()
    generate_plots(experiment_summaries, param_specific_data)
