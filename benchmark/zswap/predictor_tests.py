import matplotlib.pyplot as plt
import numpy as np
from remote_zswap_runner import RemoteZswapRunner


def set_zswap_config(runner, config_str: str) -> int:
    # If config is None, skip configuration (for zswap_off case)
    if config_str is None:
        runner.configure_zswap(parameter='enabled', value='N')
        return 0

    confs = config_str.split('_')
    if len(confs) != 8:
        print(f"String config is incorrect: {config_str} (expected 8 parameters)")
        return 1
    conf_dict = {
        'compressor': confs[0],
        'zpool': confs[1],
        'max_pool_percent': confs[2],
        'accept_threshold_percent': confs[3],
        'shrinker_enabled': confs[4],
        'exclusive_loads': confs[5],
        'same_filled_pages_enabled': confs[6],
        'non_same_filled_pages_enabled': confs[7],
    }

    # First ensure zswap is enabled
    runner.configure_zswap(parameter='enabled', value='Y')

    # Then apply all configuration parameters
    for k,v in conf_dict.items():
        runner.configure_zswap(parameter=k, value=v)
    return 0

def main():
    workload = "uniform"
    zswap_config_options = {
        'zswap_off': None,
        'default': 'lzo_zbud_20_90_Y_N_Y_Y',
        'optimal': 'lz4hc_zbud_40_100_Y_N_Y_N'
    }

    fixed_means = {}
    fixed_stds = {}
    predictor_means = {}
    predictor_stds = {}

    print("[INFO] Starting predictor vs fixed config tests...")
    runner = RemoteZswapRunner(
        remote_host='pc794.emulab.net',
        ssh_key='~/.ssh/cloudlab',
    )

    runner.establish_connection()
    '''
    runner.setup_kernmlops(owner='dariusgrassi', branch='zswap-runner', verbose=False)
    runner.reset_connection()

    # Check if inference directory exists
    inference_exists = (runner.execute_remote_command('test -d inference', get_pty=False, ignore_errors=True) == 0)
    if not inference_exists:
        print("[ERROR] Inference directory not found. Please ensure the model is properly set up.")
        return 1

    runner.setup_ycsb_experiment(benchmark='redis', distribution=workload, verbose=True)

    # Uncomment this block to execute the experiments
    for label, conf in zswap_config_options.items():
        print(f"[INFO] Testing zswap config: {label} -> {conf}")

        if label != 'zswap_off':
            set_zswap_config(runner, conf)
        else:
            # Special handling for zswap_off
            runner.configure_zswap(parameter='enabled', value='N')

        print("[INFO] Running fixed config test...")
        for i in range(5):
            runner.shrink_page_cache()
            runner.clear_swap()
            print(f"  [Fixed] Iteration {i+1}...")
            runner.run_mem_constrained_ycsb_experiment(benchmark=f"fixed_{label}_redis_{workload}")

        # Only run prediction tests for configs that aren't zswap_off
        if label != 'zswap_off':
            print("[INFO] Running predictor tests with exhaustive knob configs...")
            command = """
            tmux new-session -d -s inference 'cd inference && source .venv/bin/activate && python inference_model.py'
            """

            # Test different max_pool and accept_threshold combinations
            for max_pool in ["10", "20", "40"]:
                for accept_thresh in ["80", "90", "100"]:
                    # Create modified config string
                    conf_parts = conf.split('_')
                    conf_parts[2] = max_pool
                    conf_parts[3] = accept_thresh
                    conf_mod = '_'.join(conf_parts)

                    for i in range(5):
                        print(f"  [Predictor] {conf_mod}, iteration {i+1}")
                        set_zswap_config(runner, conf_mod)
                        runner.shrink_page_cache()
                        runner.clear_swap()
                        # Clean up any existing inference sessions
                        runner.execute_remote_command('tmux send-keys -t inference C-c', get_pty=True, ignore_errors=True)
                        runner.execute_remote_command('tmux kill-session -t inference', get_pty=False, ignore_errors=True)
                        # Start inference model
                        runner.execute_remote_command(command, get_pty=False)
                        time.sleep(2)  # Give time for the inference model to start

                        tag = f"predicted_{label}_redis_{workload}_{conf_mod}"
                        runner.run_mem_constrained_ycsb_experiment(benchmark=tag, verbose=False)

                        # Clean up the inference process
                        runner.execute_remote_command('tmux send-keys -t inference C-c', get_pty=True, ignore_errors=True)
                        runner.execute_remote_command('tmux kill-session -t inference', get_pty=False, ignore_errors=True)
                        time.sleep(1)  # Ensure clean termination

    '''
    # Parse logs and generate plot (can run independently)
    for label, conf in zswap_config_options.items():
        if label == 'zswap_off':
            # Use existing redis zswap off experiment results
            fixed_times = runner.find_and_parse_logfiles("redis_zswap_off_*")
            predictor_means[label] = None
            predictor_stds[label] = None
        else:
            if workload == 'zipfian':
                fixed_times = runner.find_and_parse_logfiles(f"fixed_{label}_redis_{workload}_*")
            else:
                fixed_times = runner.find_and_parse_logfiles(f"fixed_{label}_redis_*")

            predictor_times = []
            for max_pool in ["10", "20", "40"]:
                for accept_thresh in ["80", "90", "100"]:
                    conf_parts = conf.split('_')
                    conf_parts[2] = max_pool
                    conf_parts[3] = accept_thresh
                    conf_mod = '_'.join(conf_parts)
                    if workload == 'zipfian':
                        tag_prefix = f"predicted_{label}_redis_{workload}_{conf_mod}_*"
                    else:
                        tag_prefix = f"predicted_{label}_redis_{conf_mod}_*"
                    times = runner.find_and_parse_logfiles(tag_prefix)
                    if times and len(times) > 0:
                        predictor_times.append(times)

        if fixed_times and len(fixed_times) > 0:
            fixed_means[label] = np.mean(fixed_times) / 1000
            fixed_stds[label] = np.std(fixed_times) / 1000
        else:
            print(f"[WARNING] No fixed data found for {label}, using None")
            fixed_means[label] = None
            fixed_stds[label] = None

        if label != 'zswap_off' and predictor_times and len(predictor_times) > 0:
            avg_preds = [np.mean(p) / 1000 for p in predictor_times]
            std_preds = [np.std(p) / 1000 for p in predictor_times]
            min_idx = int(np.argmin(avg_preds))
            predictor_means[label] = avg_preds[min_idx]
            predictor_stds[label] = std_preds[min_idx]
        elif label != 'zswap_off':
            print(f"[WARNING] No predictor data found for {label}, using None")
            predictor_means[label] = None
            predictor_stds[label] = None
    # Check if we have enough valid data for plotting
    valid_predictor_values = [v for v in predictor_means.values() if v is not None]
    if not valid_predictor_values:
        print("[ERROR] No valid predictor data found, cannot create plot")
        return 1

    #print("\n[DEBUG] fixed_means:")
    #pprint(fixed_means)
    #print("\n[DEBUG] predictor_means:")
    #pprint(predictor_means)

    # New simplified bar plot with proper handling of None values
    bar_labels = ["Default", "Optimal", "Model", "Zswap Off"]

    # Safe extraction of means
    default_mean = fixed_means.get("default")
    optimal_mean = fixed_means.get("optimal")
    zswap_off_mean = fixed_means.get("zswap_off")
    model_mean = predictor_means.get("default")
    #min([v for v in predictor_means.values() if v is not None], default=np.nan)

    print(f"\n[DEBUG] Means: \nDefault={default_mean}, \nOptimal={optimal_mean}, \nModel={model_mean}, \nZswapOff={zswap_off_mean}")

    # Replace None with 0 for plotting (optional, could use np.nan too)
    bar_means = [
        default_mean if default_mean is not None else 0,
        optimal_mean if optimal_mean is not None else 0,
        model_mean if model_mean is not None else 0,
        zswap_off_mean if zswap_off_mean is not None else 0,
    ]

    # Percentage diff between model and optimal
    if optimal_mean and model_mean:
        model_pct = abs(model_mean - optimal_mean) / ((model_mean + optimal_mean) / 2)
        print(f"\nPercentage difference of optimal vs model: {model_pct * 100:.2f}%")

        default_pct = abs(model_mean - default_mean) / ((model_mean + default_mean) / 2)
        print(f"Percentage difference of default vs model: {default_pct * 100:.2f}%")

        off_pct = abs(model_mean - zswap_off_mean) / ((model_mean + zswap_off_mean) / 2)
        print(f"Percentage difference of zswap_off vs model: {off_pct * 100:.2f}%")

    # Extract stds safely
    default_std = fixed_stds.get("default")
    optimal_std = fixed_stds.get("optimal")
    zswap_off_std = fixed_stds.get("zswap_off")
    model_std = predictor_stds.get("default")

    # Handle model std carefully
    #valid_model_stds = [v for v in predictor_stds.values() if v is not None]
    #model_std_idx = np.argmin([v for v in predictor_means.values() if v is not None])
    #model_std = valid_model_stds[model_std_idx] if model_std_idx < len(valid_model_stds) else np.nan

#    print(f"""
#[DEBUG] STDs:
#    Default={default_std:.2f},
#    Optimal={optimal_std:.2f},
#    Model={model_std:.2f},
#    ZswapOff={zswap_off_std:.2f}
#    """)

    bar_stds = [
        default_std if default_std is not None else np.nan,
        optimal_std if optimal_std is not None else np.nan,
        model_std,
        zswap_off_std if zswap_off_std is not None else np.nan,
    ]
    #print("[DEBUG] bar_stds (final):")
    #pprint(bar_stds)

    # Plotting
    x = np.arange(len(bar_labels))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_ylabel('Mean Runtime (s)', fontsize=18)
    if workload == 'zipfian':
        ax.set_title('Zswap Redis Zipfian Performance Comparison', fontsize=20)
    else:
        ax.set_title('Zswap Redis Performance Comparison', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', zorder=0)

    for i, val in enumerate(bar_means):
        if val > 0 and not np.isnan(bar_stds[i]):
            ax.text(x[i], val + bar_stds[i] + 0.01, f"{val:.2f}s",
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    filename = f"predictor_vs_fixed_{workload}.png"
    plt.savefig(filename, dpi=300)
    print(f"[DONE] Plot saved as '{filename}'")

    # Print a summary of the results
    print("\nPerformance Summary:")
    print("-" * 50)
    print(f"Default Config:  {default_mean:.2f}s ± {default_std:.2f}s")
    print(f"Optimal Config:  {optimal_mean:.2f}s ± {optimal_std:.2f}s")
    print(f"Model Selected:  {model_mean:.2f}s ± {model_std:.2f}s")
    print(f"Zswap Off:      {zswap_off_mean:.2f}s ± {zswap_off_std:.2f}s")
    print("-" * 50)

    return 0

if __name__ == '__main__':
    main()
