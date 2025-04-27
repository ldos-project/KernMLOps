import numpy as np
from remote_zswap_runner import RemoteZswapRunner
from scipy import stats

ZSWAP_CONFIGS = [
    'compressor', 'zpool', 'max_pool_percent', 'accept_threshold_percent',
    'shrinker_enabled', 'exclusive_loads', 'same_filled_pages_enabled',
    'non_same_filled_pages_enabled'
]

def get_configs():
    configs = []
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

while zswap_configs:
    zswap_config = zswap_configs.pop()
    config_str = '_'.join(list(zswap_config.values()))
    runtimes = runner.find_and_parse_logfiles(config_str + '_*')
    if runtimes:
        print(config_str)
        conf_stats = calculate_statistics(runtimes)
        print(f"Mean: {conf_stats['mean']:.2f}")
        print(f"Standard Deviation: {conf_stats['std_dev']:.2f}")
        print(f"95% Confidence Interval: {conf_stats['confidence_interval_95'][0]:.2f}, {conf_stats['confidence_interval_95'][0]:.2f}")
