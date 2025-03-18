import os

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import ultraimport

# Import data import module
di = ultraimport('__dir__/../../python/kernmlops/data_import/__init__.py')

# Base directory containing all experiments
BASE_DIR = 'data/curated/'

# Define experiment renaming scheme
experiment_names = {
    "exp_a": "default",
    "exp_c": "accept_thresh",
    "exp_d": "max_pool_pct",
    "exp_e": "compressor",
    "exp_f": "zpool",
    "exp_g": "exclusive_loads ON",
    "exp_h": "non_same_filled_pages OFF",
    "exp_i": "same_filled_pages OFF",
    "exp_j": "shrinker OFF"
}

# Parameter setting display names for better readability
param_display_names = {
    # accept_thresh (exp_c)
    "thresh_40": "40% Threshold",
    "thresh_60": "60% Threshold",
    "thresh_70": "70% Threshold",

    # max_pool_pct (exp_d)
    "pool_10": "10% Pool",
    "pool_30": "30% Pool",
    "pool_40": "40% Pool",
    "pool_50": "50% Pool",
    "pool_75": "75% Pool",

    # compressor (exp_e)
    "842": "842 Compressor",
    "deflate": "Deflate Compressor",
    "lz4": "LZ4 Compressor",
    "lz4hc": "LZ4HC Compressor",
    "zstd": "ZSTD Compressor",

    # zpool (exp_f)
    "z3fold": "Z3fold Pool",
    "zsmalloc": "ZSmalloc Pool"
}

# Function to extract parameter settings from run directory names
def extract_parameter_setting(exp_id, run_dir):
    if exp_id == "exp_c":  # accept_thresh
        if "thresh_40" in run_dir:
            return "thresh_40"
        elif "thresh_60" in run_dir:
            return "thresh_60"
        elif "thresh_70" in run_dir:
            return "thresh_70"
        else:
            return "unknown_thresh"
    elif exp_id == "exp_d":  # max_pool_pct
        if "pool_10" in run_dir:
            return "pool_10"
        elif "pool_30" in run_dir:
            return "pool_30"
        elif "pool_40" in run_dir:
            return "pool_40"
        elif "pool_50" in run_dir:
            return "pool_50"
        elif "pool_75" in run_dir:
            return "pool_75"
        else:
            return "unknown_pool"
    elif exp_id == "exp_e":  # compressor
        if "842" in run_dir:
            return "842"
        elif "deflate" in run_dir:
            return "deflate"
        elif "lz4hc" in run_dir:
            return "lz4hc"
        elif "lz4" in run_dir and "hc" not in run_dir:
            return "lz4"
        elif "zstd" in run_dir:
            return "zstd"
        else:
            return "unknown_compressor"
    elif exp_id == "exp_f":  # zpool
        if "z3fold" in run_dir:
            return "z3fold"
        elif "zsmalloc" in run_dir:
            return "zsmalloc"
        else:
            return "unknown_zpool"
    else:
        return None  # Not a parameterized experiment

# Initialize dictionaries to store parameter-specific data
param_specific_data = {
    "exp_c": {},  # accept_thresh
    "exp_d": {},  # max_pool_pct
    "exp_e": {},  # compressor
    "exp_f": {}   # zpool
}

# Initialize a list to store experiment summaries
experiment_summaries = []

# For each of the plotting functions, improve the visual quality
def configure_plot_quality(fig):
    """Configure plot to have higher quality and better readability"""
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=14,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            font=dict(size=12),
            borderwidth=1
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    return fig
for exp_dir in sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]):
    exp_path = os.path.join(BASE_DIR, exp_dir)

    # Get readable name for the experiment
    exp_readable_name = experiment_names.get(exp_dir, exp_dir)
    print(f"\nProcessing experiment: {exp_dir} ({exp_readable_name})")

    # Lists to store run metrics for averaging
    collection_times = []
    zswap_durations = []
    runs_processed = 0

    # For parameterized experiments, track per-parameter data
    is_parameterized = exp_dir in ["exp_c", "exp_d", "exp_e", "exp_f"]

    # Loop through all run directories (run_1, run_2, etc.)
    for run_dir in sorted([d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]):
        run_path = os.path.join(exp_path, run_dir)
        print(f"  Processing run: {run_dir}")

        # Extract parameter setting if this is a parameterized experiment
        param_setting = None
        if is_parameterized:
            param_setting = extract_parameter_setting(exp_dir, run_dir)
            if param_setting not in param_specific_data[exp_dir]:
                param_specific_data[exp_dir][param_setting] = {
                    "collection_times": [],
                    "zswap_durations": [],
                    "runs_processed": 0
                }

        try:
            # Read parquet files for this specific run
            run_data = di.read_parquet_dir(run_path)

            # Get experiment runtime from system_info
            if 'system_info' in run_data:
                system_info = run_data['system_info']
                collection_time_sec = system_info.select(pl.col("collection_time_sec")).item()
                collection_times.append(collection_time_sec)
                print(f"    Collection time (s): {collection_time_sec}")

                # Also add to parameter-specific data if applicable
                if is_parameterized and param_setting:
                    param_specific_data[exp_dir][param_setting]["collection_times"].append(collection_time_sec)
            else:
                print(f"    Warning: No system_info found in {run_path}")
                continue

            # Get zswap runtime data
            if 'zswap_runtime' in run_data:
                zswap_data = run_data['zswap_runtime']

                # Calculate durations
                zswap_with_duration = zswap_data.with_columns(
                    (pl.col("end_ts") - pl.col("start_ts")).alias("duration_ns")
                )
                total_duration_ns = zswap_with_duration.select(pl.sum("duration_ns")).item()
                total_duration_s = total_duration_ns / 1e9
                zswap_durations.append(total_duration_s)

                # Also add to parameter-specific data if applicable
                if is_parameterized and param_setting:
                    param_specific_data[exp_dir][param_setting]["zswap_durations"].append(total_duration_s)
                    param_specific_data[exp_dir][param_setting]["runs_processed"] += 1

                print(f"    Total function calls: {zswap_data.height}")
                print(f"    Total zswap duration (s): {total_duration_s}")

                runs_processed += 1
            else:
                print(f"    Warning: No zswap_runtime found in {run_path}")
                continue

        except Exception as e:
            print(f"    Error processing {run_path}: {e}")
            continue

    # Calculate averages for this experiment
    if runs_processed > 0:
        avg_collection_time = sum(collection_times) / len(collection_times)
        avg_zswap_duration = sum(zswap_durations) / len(zswap_durations)
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

        print(f"  Summary for {exp_dir} ({exp_readable_name}): Avg Runtime: {avg_collection_time:.2f}s, Avg Zswap: {avg_zswap_duration:.2f}s ({avg_zswap_percentage:.2f}%)")
    else:
        print(f"  No valid runs processed for {exp_dir}")

# Create a summary dataframe with one row per experiment
if experiment_summaries:
    summary_df = pl.DataFrame(experiment_summaries)
    print("\nExperiment Summary DataFrame:")
    print(summary_df)

    # Filter out exp_b and exp_h from plotting
    plot_df = summary_df.filter(pl.col("experiment_id") != "exp_b").filter(pl.col("experiment_id") != "exp_h")

    # Create bar chart comparing experiments
    fig1 = px.bar(
        plot_df.to_pandas(),
        x="experiment",
        y="avg_runtime_s",
        title="Average Kernel Benchmark Runtime by Zswap Configuration",
        labels={
            "avg_runtime_s": "Average Runtime (seconds)",
            "experiment": "Zswap Configuration"
        },
        text_auto='.2f',
        height=600,
        width=900
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # Create bar chart for zswap duration
    fig2 = px.bar(
        plot_df.to_pandas(),
        x="experiment",
        y="avg_zswap_duration_s",
        title="Average Time Spent in Zswap Functions by Configuration",
        labels={
            "avg_zswap_duration_s": "Average Zswap Function Duration (seconds)",
            "experiment": "Zswap Configuration"
        },
        text_auto='.2f',
        height=600,
        width=900
    )
    fig2.update_layout(xaxis_tickangle=-45)

    # Create a percentage chart
    fig3 = px.bar(
        plot_df.to_pandas(),
        x="experiment",
        y="avg_zswap_percentage",
        title="Percentage of Runtime Spent in Zswap Functions by Configuration",
        labels={
            "avg_zswap_percentage": "Zswap Function Time (%)",
            "experiment": "Zswap Configuration"
        },
        text_auto='.2f',
        height=600,
        width=900
    )
    fig3.update_layout(xaxis_tickangle=-45)

    # For each of the parameterized experiment types, create dedicated plots
    for exp_id in ["exp_c", "exp_d", "exp_e", "exp_f"]:
        if not param_specific_data[exp_id]:
            continue

        # Calculate averages for each parameter setting
        param_summaries = []
        for param, data in param_specific_data[exp_id].items():
            if data["runs_processed"] > 0:
                avg_runtime = sum(data["collection_times"]) / len(data["collection_times"])
                avg_zswap = sum(data["zswap_durations"]) / len(data["zswap_durations"])
                avg_percentage = (avg_zswap / avg_runtime * 100) if avg_runtime > 0 else 0

                param_summaries.append({
                    "param_id": param,
                    "param_name": param_display_names.get(param, param),
                    "avg_runtime_s": avg_runtime,
                    "avg_zswap_duration_s": avg_zswap,
                    "avg_zswap_percentage": avg_percentage,
                    "runs_processed": data["runs_processed"]
                })

        if param_summaries:
            # Create a param-specific DataFrame
            param_df = pl.DataFrame(param_summaries)

            # Sort by parameter ID for consistent ordering
            param_df = param_df.sort("param_id")

            # Get experiment readable name
            exp_name = experiment_names.get(exp_id, exp_id)

            # Create parameter-specific runtime plot
            param_fig1 = px.bar(
                param_df.to_pandas(),
                x="param_name",
                y="avg_runtime_s",
                title=f"Average Runtime by {exp_name} Setting",
                labels={
                    "avg_runtime_s": "Average Runtime (seconds)",
                    "param_name": f"{exp_name} Setting"
                },
                text_auto='.2f',
                height=600,
                width=900
            )
            param_fig1.update_layout(xaxis_tickangle=-45)

            # Create parameter-specific zswap duration plot
            param_fig2 = px.bar(
                param_df.to_pandas(),
                x="param_name",
                y="avg_zswap_duration_s",
                title=f"Average Zswap Time by {exp_name} Setting",
                labels={
                    "avg_zswap_duration_s": "Average Zswap Duration (seconds)",
                    "param_name": f"{exp_name} Setting"
                },
                text_auto='.2f',
                height=600,
                width=900
            )
            param_fig2.update_layout(xaxis_tickangle=-45)

            # Create parameter-specific percentage plot
            param_fig3 = px.bar(
                param_df.to_pandas(),
                x="param_name",
                y="avg_zswap_percentage",
                title=f"Percentage of Runtime in Zswap by {exp_name} Setting",
                labels={
                    "avg_zswap_percentage": "Zswap Time (%)",
                    "param_name": f"{exp_name} Setting"
                },
                text_auto='.2f',
                height=600,
                width=900
            )
            param_fig3.update_layout(xaxis_tickangle=-45)

            # Create a combined plot for this parameter
            param_fig4 = go.Figure()

            # Add traces for runtime and zswap duration
            param_fig4.add_trace(go.Bar(
                x=param_df.to_pandas()["param_name"],
                y=param_df.to_pandas()["avg_runtime_s"],
                name="Total Runtime",
                marker_color='lightblue'
            ))

            param_fig4.add_trace(go.Bar(
                x=param_df.to_pandas()["param_name"],
                y=param_df.to_pandas()["avg_zswap_duration_s"],
                name="Zswap Duration",
                marker_color='darkblue'
            ))

            param_fig4.update_layout(
                title=f"Runtime vs Zswap Duration by {exp_name} Setting",
                xaxis_title=f"{exp_name} Setting",
                yaxis_title="Duration (seconds)",
                xaxis_tickangle=-45,
                barmode='group',
                height=700,
                width=1000,
                legend=dict(x=0.01, y=0.99)
            )

            # Apply quality enhancements
            configure_plot_quality(param_fig1)
            configure_plot_quality(param_fig2)
            configure_plot_quality(param_fig3)
            configure_plot_quality(param_fig4)

            # Save parameter-specific plots
            param_fig1.write_image(f"images/zswap_{exp_name}_runtime_comparison.png")
            param_fig2.write_image(f"images/zswap_{exp_name}_function_duration.png")
            param_fig3.write_image(f"images/zswap_{exp_name}_percentage_comparison.png")
            param_fig4.write_image(f"images/zswap_{exp_name}_runtime_vs_duration.png")

            # Export parameter summary to CSV
            param_df.write_csv(f"zswap_{exp_name}_performance_summary.csv")

    # Create the combined figure
    fig4 = go.Figure()

    fig4.add_trace(go.Bar(
        x=plot_df.to_pandas()["experiment"],
        y=plot_df.to_pandas()["avg_runtime_s"],
        name="Total Runtime",
        marker_color='lightblue'
    ))

    fig4.add_trace(go.Bar(
        x=plot_df.to_pandas()["experiment"],
        y=plot_df.to_pandas()["avg_zswap_duration_s"],
        name="Zswap Duration",
        marker_color='darkblue'
    ))

    fig4.update_layout(
        title="Kernel Benchmark Runtime vs Zswap Function Duration by Configuration",
        xaxis_title="Zswap Configuration",
        yaxis_title="Duration (seconds)",
        xaxis_tickangle=-45,
        barmode='group',
        height=700,
        width=1000,
        legend=dict(x=0.01, y=0.99)
    )

    # Apply quality enhancements to all figures
    configure_plot_quality(fig1)
    configure_plot_quality(fig2)
    configure_plot_quality(fig3)
    configure_plot_quality(fig4)

    # Make sure images directory exists
    os.makedirs("images", exist_ok=True)

    # Uncomment to show the combined plot
    # fig4.show()

    # Create a directory for PDFs if it doesn't exist
    os.makedirs("pdfs", exist_ok=True)

    # Save the plots with descriptive filenames as high-quality PDFs
    fig1.write_image("pdfs/zswap_config_runtime_comparison.pdf", scale=2)
    fig2.write_image("pdfs/zswap_config_function_duration.pdf", scale=2)
    fig3.write_image("pdfs/zswap_config_percentage_comparison.pdf", scale=2)
    fig4.write_image("pdfs/zswap_config_runtime_vs_function_duration.pdf", scale=2)

    # Also save as PNGs for quick viewing if needed
    fig1.write_image("images/zswap_config_runtime_comparison.png", scale=2)
    fig2.write_image("images/zswap_config_function_duration.png", scale=2)
    fig3.write_image("images/zswap_config_percentage_comparison.png", scale=2)
    fig4.write_image("images/zswap_config_runtime_vs_function_duration.png", scale=2)

    # Export summary to CSV with descriptive filename
    summary_df.write_csv("zswap_configuration_performance_summary.csv")
else:
    print("No experiment data was processed successfully")
