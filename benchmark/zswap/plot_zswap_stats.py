import re

import matplotlib.pyplot as plt
import numpy as np
from remote_zswap_runner import RemoteZswapRunner
from scipy import stats

ZSWAP_CONFIGS = [
    'workloads', 'compressor', 'zpool', 'max_pool_percent',
    'accept_threshold_percent', 'shrinker_enabled', 'exclusive_loads',
    'same_filled_pages_enabled', 'non_same_filled_pages_enabled'
]

def get_configs():
    configs = []
    for workloads in ['redis', 'mongodb']:
        configs.append({'workloads': workloads, 'zswap_off': 'zswap_off'})
        for compressor in ['lzo', 'deflate', '842', 'lz4', 'lz4hc', 'zstd']:
            for zpool in ['zbud', 'z3fold', 'zsmalloc']:
                for max_pool_percent in ['10', '20', '40']:
                    for accept_threshold_percent in ['80', '90', '100']:
                        for shrinker_enabled in ['Y','N']:
                            for exclusive_loads in ['Y','N']:
                                for same_filled_pages_enabled in ['Y','N']:
                                    for non_same_filled_pages_enabled in ['Y','N']:
                                        config = {name: value
                                            for name,value in locals().items()
                                            if name in ZSWAP_CONFIGS}
                                        configs.append(config)
    return configs

def calculate_statistics(data):
    data_s = []
    for d in data:
        data_s.append(d / 1000.0)
    mean = np.mean(data_s)
    std_dev = np.std(data_s, ddof=1)
    n = len(data_s)
    # Calculate 95% confidence interval for the mean
    # Using t-distribution for small samples
    t_crit = stats.t.ppf(0.975, n-1)  # 95% CI requires 0.975 (two-tailed)
    std_err = std_dev / np.sqrt(n)
    ci_lower = mean - t_crit * std_err
    ci_upper = mean + t_crit * std_err
    return {
        "mean": mean,
        "std_dev": std_dev,
        "sample_size": n,
        "confidence_interval_95": (ci_lower, ci_upper)
    }

def plot_filtered_bars(conf_names, conf_means, conf_stds, is_zswap_off, title, ylabel, filename, highlight_config=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    normal_bars = [i for i, is_off in enumerate(is_zswap_off) if not is_off]
    off_bars = [i for i, is_off in enumerate(is_zswap_off) if is_off]
    ax.bar([conf_names[i] for i in normal_bars],
           [conf_means[i] for i in normal_bars],
           label='Normal Config')
    ax.bar([conf_names[i] for i in off_bars],
           [conf_means[i] for i in off_bars],
           color='red',
           edgecolor='black',
           linewidth=2,
           hatch='///',
           alpha=1.0,
           label='zswap_off Config')
    for i in off_bars:
        ax.annotate(f'ZSWAP OFF\n{conf_means[i]:.2f}s',
                   xy=(i, conf_means[i] + 0.1),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   color='red',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    min_index = conf_means.index(min(conf_means))
    ax.bar(conf_names[min_index], conf_means[min_index],
           color='green',
           edgecolor='black',
           linewidth=2,
           alpha=0.7)
    ax.annotate(f'{conf_names[min_index]}\n{conf_means[min_index]:.2f}s',
               xy=(min_index, conf_means[min_index]),
               xytext=(0, -35),
               textcoords='offset points',
               ha='center',
               va='top',
               fontsize=8,
               color='green',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    if highlight_config and highlight_config in conf_names:
        i = conf_names.index(highlight_config)
        ax.bar(highlight_config, conf_means[i],
               color='purple',
               edgecolor='black',
               linewidth=2,
               alpha=0.7)
        ax.annotate(f'{highlight_config}\n{conf_means[i]:.2f}s',
                   xy=(i, conf_means[i]),
                   xytext=(0, 35),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=8,
                   color='purple',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_selected_configs(conf_names, conf_means, conf_stds, is_zswap_off, default_config, title, ylabel, filename, alt_config=None, alt_config_name="alt config"):
    selected = []
    labels = []
    colors = []
    min_index = conf_means.index(min(conf_means))
    print("Optimal config:", conf_names[min_index])
    if default_config in conf_names:
        default_index = conf_names.index(default_config)
        selected.append(default_index)
        labels.append("default")
        colors.append("#9e9ac8")  # soft purple
    selected.append(min_index)
    labels.append("optimal")
    colors.append("#74c476")  # soft green
    if alt_config and alt_config in conf_names:
        alt_index = conf_names.index(alt_config)
        selected.append(alt_index)
        labels.append(alt_config_name)
        colors.append("#6baed6")  # soft blue
    for i, name in enumerate(conf_names):
        if is_zswap_off[i]:
            selected.append(i)
            labels.append("zswap off")
            colors.append("#fc9272")  # soft red
    selected_means = [conf_means[i] for i in selected]
    selected_stds = [conf_stds[i] for i in selected]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, selected_means, yerr=selected_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for i, val in enumerate(selected_means):
        ax.text(i, val + 0.05, f"{val:.2f}s", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_predicted_configs(conf_names, conf_means, conf_stds, is_zswap_off, default_config, alt_config, alt_config_name, title, ylabel, filename):
    selected = []
    labels = []
    colors = []

    # Default config
    if default_config in conf_names:
        idx = conf_names.index(default_config)
        selected.append(idx)
        labels.append("default")
        colors.append("#9e9ac8")

    # Optimal (overall best)
    min_idx = conf_means.index(min(conf_means))
    selected.append(min_idx)
    labels.append("optimal")
    colors.append("#74c476")

    # Regex match for alt_config and pick the best among them
    regex = re.compile('^' + alt_config.replace('*', '.*') + '$')
    matching_idxs = [i for i, name in enumerate(conf_names) if regex.match(name)]
    print(matching_idxs)
    if matching_idxs:
        best_alt_idx = min(matching_idxs, key=lambda i: conf_means[i])
        selected.append(best_alt_idx)
        labels.append(alt_config_name)
        colors.append("#6baed6")

    # Zswap off configs
    for i, is_off in enumerate(is_zswap_off):
        if is_off:
            selected.append(i)
            labels.append("zswap off")
            colors.append("#fc9272")

    # Plot
    selected_means = [conf_means[i] for i in selected]
    selected_stds = [conf_stds[i] for i in selected]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, selected_means, yerr=selected_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for i, val in enumerate(selected_means):
        ax.text(i, val + 0.05, f"{val:.2f}s", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

zswap_configs = get_configs()
runner = RemoteZswapRunner(
    remote_host='dwg@whatever',
    ssh_key='~/.ssh/cloudlab'
)

# Process data and track which configs are zswap_off
redis_conf_names = []
redis_conf_means = []
redis_conf_stds = []
redis_is_zswap_off = []
mongodb_conf_names = []
mongodb_conf_means = []
mongodb_conf_stds = []
mongodb_is_zswap_off = []
while zswap_configs:
    zswap_config = list(zswap_configs.pop().values())
    config_str = '_'.join(zswap_config)
    runtimes = runner.find_and_parse_logfiles(config_str + '_*')
    is_zswap_off = 'zswap_off' in config_str
    if runtimes:
        conf_stats = calculate_statistics(runtimes)
        if zswap_config[0] == 'redis':
            redis_conf_names.append(config_str)
            redis_conf_means.append(conf_stats['mean'])
            redis_conf_stds.append(conf_stats['std_dev'])
            redis_is_zswap_off.append(is_zswap_off)
        elif zswap_config[0] == 'mongodb':
            mongodb_conf_names.append(config_str)
            mongodb_conf_means.append(conf_stats['mean'])
            mongodb_conf_stds.append(conf_stats['std_dev'])
            mongodb_is_zswap_off.append(is_zswap_off)

'''
plot_selected_configs(
    redis_conf_names,
    redis_conf_means,
    redis_conf_stds,
    redis_is_zswap_off,
    default_config='redis_lzo_zbud_20_90_Y_N_Y_Y',
    alt_config='redis_lz4hc_zbud_40_100_Y_N_Y_N',
    alt_config_name='uniform optimal',
    title='YCSB Redis Benchmark (Zipfian Distribution) Zswap Default Config vs. Optimal vs. Uniform Optimal vs. OFF',
    ylabel='Runtime (s)',
    filename='zipfian_redis_selected.png'
)
'''

# TODO: plot wildcarded predicted outcomes
plot_predicted_configs(
    redis_conf_names,
    redis_conf_means,
    redis_conf_stds,
    redis_is_zswap_off,
    default_config='redis_lzo_zbud_20_90_Y_N_Y_Y',
    alt_config='redis_deflate_zsmalloc_40_80_Y_N_N_Y',
    alt_config_name='predicted optimal',
    title='YCSB Redis (Uniform) Zswap Config Default vs. Optimal vs. Predicted Optimal vs. OFF',
    ylabel='Runtime (s)',
    filename='redis_uniform_predicted_selected2.png'
)

'''
# Redis plot
if redis_conf_means:
    fig, ax = plt.subplots(figsize=(12, 8))
    normal_bars = [i for i, is_off in enumerate(redis_is_zswap_off) if not is_off]
    off_bars = [i for i, is_off in enumerate(redis_is_zswap_off) if is_off]
    # normal configs
    ax.bar([redis_conf_names[i] for i in normal_bars],
           [redis_conf_means[i] for i in normal_bars],
           label='Normal Config')
    # zswap_off configs
    zswap_off_bars = ax.bar([redis_conf_names[i] for i in off_bars],
                          [redis_conf_means[i] for i in off_bars],
                          color='red',
                          edgecolor='black',
                          linewidth=2,
                          hatch='///',
                          alpha=1.0,
                          label='zswap_off Config')
    # zswap_off annotations
    for i in off_bars:
        runtime_value = redis_conf_means[i]
        ax.annotate(f'ZSWAP OFF\n{runtime_value:.2f}s',
                   xy=(i, redis_conf_means[i] + 0.1),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   color='red',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    # minimum runtime annotation
    min_index = redis_conf_means.index(min(redis_conf_means))
    min_config_name = redis_conf_names[min_index]
    min_runtime = redis_conf_means[min_index]
    ax.bar(min_config_name, min_runtime,
           color='green',
           edgecolor='black',
           linewidth=2,
           alpha=0.7)
    ax.annotate(f'{min_config_name}\n{min_runtime:.2f}s',
                xy=(min_index, min_runtime),
                xytext=(0, -35),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=8,
                color='green',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    # default config annotation: redis_lzo_zbud_20_90_Y_N_Y_Y
    specific_config = 'redis_lzo_zbud_20_90_Y_N_Y_Y'
    if specific_config in redis_conf_names:
        specific_index = redis_conf_names.index(specific_config)
        specific_runtime = redis_conf_means[specific_index]
        ax.bar(specific_config, specific_runtime,
               color='purple',
               edgecolor='black',
               linewidth=2,
               alpha=0.7)
        ax.annotate(f'{specific_config}\n{specific_runtime:.2f}s',
                   xy=(specific_index, specific_runtime),
                   xytext=(0, 35),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=8,
                   color='purple',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    else:
        print(f"Config '{specific_config}' not found in the data")
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylabel('Redis Mean Runtimes (s)', fontsize=12)
    ax.set_title('Redis Configuration Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('redis_conf_enhanced.png', dpi=300)

# Mongodb plot
if mongodb_conf_means:
    fig, ax = plt.subplots(figsize=(12, 8))
    normal_bars = [i for i, is_off in enumerate(mongodb_is_zswap_off) if not is_off]
    off_bars = [i for i, is_off in enumerate(mongodb_is_zswap_off) if is_off]

    ax.bar([mongodb_conf_names[i] for i in normal_bars],
           [mongodb_conf_means[i] for i in normal_bars],
           label='Normal Config')
    ax.bar([mongodb_conf_names[i] for i in off_bars],
           [mongodb_conf_means[i] for i in off_bars],
           color='red',
           edgecolor='black',
           linewidth=2,
           hatch='///',
           alpha=1.0,
           label='zswap_off Config')

    for i in off_bars:
        ax.annotate(f'ZSWAP OFF\n{mongodb_conf_means[i]:.2f}s',
                   xy=(i, mongodb_conf_means[i] + 0.1),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   color='red',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    min_index = mongodb_conf_means.index(min(mongodb_conf_means))
    min_config_name = mongodb_conf_names[min_index]
    min_runtime = mongodb_conf_means[min_index]
    ax.bar(min_config_name, min_runtime,
           color='green',
           edgecolor='black',
           linewidth=2,
           alpha=0.7)
    ax.annotate(f'optimal\n{min_config_name}\n{min_runtime:.2f}s',
               xy=(min_index, min_runtime),
               xytext=(0, -45),
               textcoords='offset points',
               ha='center',
               va='top',
               fontsize=8,
               color='green',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    default_config = 'mongodb_lzo_zbud_20_90_Y_N_Y_Y'
    if default_config in mongodb_conf_names:
        i = mongodb_conf_names.index(default_config)
        rt = mongodb_conf_means[i]
        ax.bar(default_config, rt,
               color='purple',
               edgecolor='black',
               linewidth=2,
               alpha=0.7)
        ax.annotate(f'default config\n{rt:.2f}s',
                   xy=(i, rt),
                   xytext=(0, 65),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=8,
                   color='purple',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    else:
        print(f"Default config '{default_config}' not found")

    uniform_optimal = 'mongodb_lzo_zbud_10_90_N_N_Y_N'
    if uniform_optimal in mongodb_conf_names:
        i = mongodb_conf_names.index(uniform_optimal)
        rt = mongodb_conf_means[i]
        ax.bar(uniform_optimal, rt,
               color='#6baed6',
               edgecolor='black',
               linewidth=2,
               alpha=0.7)
        ax.annotate(f'uniform optimal\n{rt:.2f}s',
                   xy=(i, rt),
                   xytext=(0, 25),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   fontsize=8,
                   color='blue',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    else:
        print(f"Uniform optimal config '{uniform_optimal}' not found")

    ax.legend(fontsize=12, loc='upper right')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylabel('MongoDB Mean Runtimes (s)', fontsize=12)
    ax.set_title('MongoDB Configuration Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('mongodb_conf_enhanced.png', dpi=300)

# Redis with compressor=lzo
if redis_conf_means:
    indices = [i for i, name in enumerate(redis_conf_names) if '_lzo_' in name]
    if indices:
        plot_filtered_bars(
            [redis_conf_names[i] for i in indices],
            [redis_conf_means[i] for i in indices],
            [redis_conf_stds[i] for i in indices],
            [redis_is_zswap_off[i] for i in indices],
            title='Redis Configs with Compressor=LZO',
            ylabel='Redis Mean Runtimes (s)',
            filename='redis_lzo.png'
        )

# Redis with compressor=lz4hc
if redis_conf_means:
    indices = [i for i, name in enumerate(redis_conf_names) if '_lz4hc_' in name]
    if indices:
        plot_filtered_bars(
            [redis_conf_names[i] for i in indices],
            [redis_conf_means[i] for i in indices],
            [redis_conf_stds[i] for i in indices],
            [redis_is_zswap_off[i] for i in indices],
            title='Redis Configs with Compressor=LZ4HC',
            ylabel='Redis Mean Runtimes (s)',
            filename='redis_lz4hc.png'
        )

# Redis with zpool=zbud
if redis_conf_means:
    indices = [i for i, name in enumerate(redis_conf_names) if '_zbud_' in name]
    if indices:
        plot_filtered_bars(
            [redis_conf_names[i] for i in indices],
            [redis_conf_means[i] for i in indices],
            [redis_conf_stds[i] for i in indices],
            [redis_is_zswap_off[i] for i in indices],
            title='Redis Configs with Zpool=ZBUD',
            ylabel='Redis Mean Runtimes (s)',
            filename='redis_zbud.png'
        )

# MongoDB with compressor=lzo
if mongodb_conf_means:
    indices = [i for i, name in enumerate(mongodb_conf_names) if '_lzo_' in name]
    if indices:
        plot_filtered_bars(
            [mongodb_conf_names[i] for i in indices],
            [mongodb_conf_means[i] for i in indices],
            [mongodb_conf_stds[i] for i in indices],
            [mongodb_is_zswap_off[i] for i in indices],
            title='MongoDB Configs with Compressor=LZO',
            ylabel='MongoDB Mean Runtimes (s)',
            filename='mongodb_lzo.png'
        )

# MongoDB with compressor=lz4hc
if mongodb_conf_means:
    indices = [i for i, name in enumerate(mongodb_conf_names) if '_lz4hc_' in name]
    if indices:
        plot_filtered_bars(
            [mongodb_conf_names[i] for i in indices],
            [mongodb_conf_means[i] for i in indices],
            [mongodb_conf_stds[i] for i in indices],
            [mongodb_is_zswap_off[i] for i in indices],
            title='MongoDB Configs with Compressor=LZ4HC',
            ylabel='MongoDB Mean Runtimes (s)',
            filename='mongodb_lz4hc.png'
        )

# MongoDB with zpool=zbud
if mongodb_conf_means:
    indices = [i for i, name in enumerate(mongodb_conf_names) if '_zbud_' in name]
    if indices:
        plot_filtered_bars(
            [mongodb_conf_names[i] for i in indices],
            [mongodb_conf_means[i] for i in indices],
            [mongodb_conf_stds[i] for i in indices],
            [mongodb_is_zswap_off[i] for i in indices],
            title='MongoDB Configs with Zpool=ZBUD',
            ylabel='MongoDB Mean Runtimes (s)',
            filename='mongodb_zbud.png'
        )
'''
