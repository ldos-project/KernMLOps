import argparse
import os

import matplotlib.pyplot as plt
from single_exp_stats import ExperimentAnalyzer


def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Run experiments with configurable parameters')
    parser.add_argument('-b', '--benchmark', type=str, choices=['redis', 'mongodb', 'gap'],
                        default='redis', help='Benchmark type (default: redis)')
    parser.add_argument('-d', '--dist', type=str, choices=['uniform', 'zipfian'],
                        default='uniform', help='Distribution type (default: uniform)')
    parser.add_argument('-t', '--thp', action='store_true',
                        help='Enable THP setting (if flag is present, thp="thp", otherwise empty string)')
    args = parser.parse_args()

    benchmark = args.benchmark
    dist = args.dist
    thp = 'thp' if args.thp else ''

    experiments = [
        'off',
        'off_mem',
        'on',
        'accept_thresh',
        'max_pool_pct',
        'compressor',
        'zpool',
        'exclusive_loads_on',
        'non_same_filled_pages_off',
        'same_filled_pages_off',
        # 'shrinker_off',
        # 'cpu_tuning'
    ]

    collection_times = []
    error_bars = []
    zswap_times = []
    zswap_error_bars = []
    param_experiment_data = {}

    analyzer = ExperimentAnalyzer()
    for experiment in experiments:
        if thp:
            exp_name = f"{benchmark}_{dist}_{thp}_zswap_{experiment}"
            print(f"\nExperiment {exp_name}")
        else:
            exp_name = f"{benchmark}_{dist}_zswap_{experiment}"
            print(f"\nExperiment {exp_name}")

        # Initialize nested dictionary for this experiment
        param_experiment_data[exp_name] = {}

        results = analyzer.analyze_experiment(
            benchmark,
            thp,
            experiment,
            dist=dist,
            verbose=False
        )
        # Collect mean completion time between runs with CI
        ct = results['collection_times']
        if ct['count'] > 0:
            mean = ct.get('mean')
            ci = ct.get('confidence_interval')
            print(f"Average: {mean:.2f}s")
            if ci and len(ct['values']) > 1:
                print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]s")
                error = (ci[1] - ci[0]) / 2
            else:
                error = 0
            collection_times.append(mean)
            error_bars.append(error)
        # Collect mean zswap runtime between runs with CI
        zs = results['zswap_durations']
        if zs['count'] > 0:
            zmean = zs.get('mean')
            zci = zs.get('confidence_interval')
            print(f"ZSwap Mean: {zmean:.2f}s")
            if zci and len(zs['values']) > 1:
                print(f"95% CI: [{zci[0]:.2f}, {zci[1]:.2f}]s")
                zerror = (zci[1] - zci[0]) / 2
            else:
                zerror = 0
        else:
            zmean = 0
            zerror = 0
        zswap_times.append(zmean)
        zswap_error_bars.append(zerror)
        # Collect per-param means and CI for applicable experiments
        try:
            for value in results['param_values']:
                ps = results['param_values'][value]['collection_times']
                if ps['count'] > 0:
                    print(value)
                    pmean = ps.get('mean')
                    print(f"    Mean: {pmean:.2f}s")
                    pci = ps.get('confidence_interval')
                    if pci and len(ps['values']) > 1:
                        print(f"    95% CI: [{pci[0]:.2f}, {pci[1]:.2f}]s")
                        perror = (pci[1] - pci[0]) / 2
                    else:
                        perror = 0
                    # Store in the param_experiment_data dictionary
                    param_experiment_data[exp_name][str(value)] = {
                        'mean': pmean,
                        'error': perror,
                    }
                else:
                    pmean = 0
                    perror = 0
                    param_experiment_data[exp_name][str(value)] = {
                        'mean': pmean,
                        'error': perror,
                    }
        except Exception:
            continue

    # Visualize param-level experiment results
    for exp_name, params in param_experiment_data.items():
        if params:  # If there are parameters for this experiment
            plt.figure(figsize=(10, 5))
            param_names = list(params.keys())
            param_means = [params[p]['mean'] for p in param_names]
            param_errors = [params[p]['error'] for p in param_names]
            plt.bar(param_names, param_means, yerr=param_errors, capsize=5,
                    ecolor='black', alpha=0.75)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Parameter Value')
            plt.ylabel('Collection Time (sec)')
            plt.title(f"Parameter comparison for {exp_name}")
            plt.tight_layout()
            os.makedirs("images/params", exist_ok=True)
            plt.savefig(f"images/params/{exp_name}_params.png")

    # Create chart for overall collection times
    pretty_experiments = [
        'Zswap OFF, no cgroup',
        'Zswap OFF, w/ cgroup 2G',
        'Zswap ON, w/ cgroup 2G, defaults',
        'Tuned % Accept Threshold',
        'Tuned Max Pool %',
        'Tuned Compressor',
        'Tuned Zpool',
        'Exclusive Loads ON',
        'Non-Same-Filled Pages OFF',
        'Same-Filled Pages OFF',
        'Shrinker OFF',
        'Tuned # CPUs'
    ]
    plt.figure(figsize=(12,6))

    plt.bar(pretty_experiments, collection_times, yerr=error_bars, capsize=5,
            ecolor='black', alpha=0.75, label='Benchmark Runtime')
    plt.bar(pretty_experiments, zswap_times, yerr=zswap_error_bars, capsize=5,
            ecolor='black', alpha=0.75, label='Zswap Wall Time')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Zswap Experiment Type')
    plt.ylabel('Collection Time (sec)')
    has_thp = 'Has' if thp else 'No'
    plt.title(f"{args.dist} {args.benchmark} ({has_thp} THP) Completion Time w/ Tuned Zswap Parameters")
    plt.legend(loc='upper right')
    plt.ylim(bottom=0, top=plt.ylim()[1]+50)
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{args.benchmark}_{args.dist}_{has_thp.lower()}_thp.png")


if __name__ == "__main__":
    main()
