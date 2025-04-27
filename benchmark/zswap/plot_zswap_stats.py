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

zswap_configs = get_configs()
runner = RemoteZswapRunner(
    remote_host='dwg@whatever',
    ssh_key='~/.ssh/cloudlab'
)

# TODO: only process for redis then make a separate graph for mongodb :p
redis_conf_names = []
redis_conf_means = []
redis_conf_stds = []
mongodb_conf_names = []
mongodb_conf_means = []
mongodb_conf_stds = []
while zswap_configs:
    zswap_config = list(zswap_configs.pop().values())
    config_str = '_'.join(zswap_config)
    runtimes = runner.find_and_parse_logfiles(config_str + '_*')
    if runtimes:
        print(config_str)
        conf_stats = calculate_statistics(runtimes)
        print(f"Mean: {conf_stats['mean']:.2f}")
        print(f"Standard Deviation: {conf_stats['std_dev']:.2f}")
        print(f"95% Confidence Interval: {conf_stats['confidence_interval_95'][0]:.2f}, {conf_stats['confidence_interval_95'][0]:.2f}")
        if zswap_config[0] == 'redis':
            redis_conf_names.append(config_str)
            redis_conf_means.append(conf_stats['mean'])
            redis_conf_stds.append(conf_stats['std_dev'])
        elif zswap_config[0] == 'mongodb':
            mongodb_conf_names.append(config_str)
            mongodb_conf_means.append(conf_stats['mean'])
            mongodb_conf_stds.append(conf_stats['std_dev'])

# Redis plot
if redis_conf_means:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(redis_conf_names, redis_conf_means, yerr=redis_conf_stds, capsize=1, error_kw={'elinewidth': 0.8, 'capthick': 0.8})
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylabel('Redis Mean Runtimes')
    plt.tight_layout()
    # plt.show()
    plt.savefig('redis_conf.png')

# Mongodb plot
if mongodb_conf_means:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(mongodb_conf_names, mongodb_conf_means, yerr=mongodb_conf_stds, capsize=1, error_kw={'elinewidth': 0.8, 'capthick': 0.8})
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylabel('Mongodb Mean Runtimes')
    plt.tight_layout()
    # plt.show()
    plt.savefig('mongodb_conf.png')
