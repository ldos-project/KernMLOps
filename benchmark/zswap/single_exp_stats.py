#!/usr/bin/env python3
import glob
import os

import numpy as np
import polars as pl
import scipy.stats as st
import ultraimport


class ExperimentAnalyzer:
    def __init__(self, base_dir='data/curated'):
        self.base_dir = base_dir
        try:
            self.data_import = ultraimport('__dir__/../../python/kernmlops/data_import/__init__.py')
            if not hasattr(self.data_import, 'read_parquet_dir'):
                self.data_import = None
        except Exception:
            self.data_import = None

    def analyze_experiment(self, benchmark_name, thp_config, experiment_name, dist='uniform', verbose=False):
        if self.data_import is None:
            return None

        # Construct experiment directory based on format
        if thp_config:
            experiment_dir = f"{benchmark_name}_{dist}_{thp_config}_{experiment_name}"
        else:
            experiment_dir = f"{benchmark_name}_{dist}_{experiment_name}"

        print(experiment_dir)
        # Define parameter categories to analyze
        param_categories = {
            'zswap_accept_thresh': [40, 60, 70, 80, 100],
            'zswap_compressor': ['842', 'deflate', 'lz4', 'lz4hc', 'zstd'],
            'zswap_cpu_tuning': [2, 4, 8, 16, 32],
            'zswap_max_pool_pct': [10, 30, 40, 50, 75],
            'zswap_zpool': ['z3fold', 'zsmalloc']
        }

        path_pattern = os.path.join(
            self.base_dir, experiment_dir, "*run_*"
        )

        run_dirs = sorted(glob.glob(path_pattern))

        if not run_dirs:
            return None

        collection_times = []
        zswap_durations_sec = []

        # Check if this is a parameter experiment
        param_specific_analysis = experiment_name in param_categories
        param_values = {}

        if param_specific_analysis:
            # Initialize data structures for parameter-specific analysis
            param_values = {value: {'collection_times': [], 'zswap_durations': []}
                           for value in param_categories[experiment_name]}

        for run_dir in run_dirs:
            run_data = self._process_run(run_dir, verbose)
            if run_data:
                # Add to cumulative results
                if 'collection_time' in run_data and run_data['collection_time'] is not None:
                    collection_times.append(run_data['collection_time'])
                if 'zswap_duration' in run_data and run_data['zswap_duration'] is not None:
                    zswap_durations_sec.append(run_data['zswap_duration'])

                # If this is a parameter experiment, add to specific parameter value
                if param_specific_analysis:
                    run_basename = os.path.basename(run_dir)

                    # Extract the parameter value from the run directory name
                    for value in param_categories[experiment_name]:
                        value_str = str(value)

                        # Match pattern depends on the experiment type
                        match_str = ''
                        if experiment_name == 'zswap_accept_thresh':
                            match_str = f"thresh_{value_str}"
                        elif experiment_name == 'zswap_compressor':
                            match_str = f"{value_str}_"
                        elif experiment_name == 'zswap_cpu_tuning':
                            match_str = f"cpus_{value_str}"
                        elif experiment_name == 'zswap_max_pool_pct':
                            match_str = f"pool_{value_str}"
                        elif experiment_name == 'zswap_zpool':
                            match_str = f"{value_str}_"

                        if match_str in run_basename:
                            if 'collection_time' in run_data and run_data['collection_time'] is not None:
                                param_values[value]['collection_times'].append(run_data['collection_time'])
                            if 'zswap_duration' in run_data and run_data['zswap_duration'] is not None:
                                param_values[value]['zswap_durations'].append(run_data['zswap_duration'])
            # break

        # Calculate cumulative statistics
        results = {
            'total_runs': len(run_dirs),
            'collection_times': {
                'count': len(collection_times),
                'values': collection_times
            },
            'zswap_durations': {
                'count': len(zswap_durations_sec),
                'values': zswap_durations_sec
            }
        }

        if collection_times:
            coll_mean, coll_total, coll_ci, coll_min = self._calculate_stats(collection_times)
            results['collection_times'].update({
                'mean': coll_mean,
                'total': coll_total,
                'confidence_interval': coll_ci,
                'min': coll_min
            })

        if zswap_durations_sec:
            zswap_mean, zswap_total, zswap_ci, zswap_min = self._calculate_stats(zswap_durations_sec)
            results['zswap_durations'].update({
                'mean': zswap_mean,
                'total': zswap_total,
                'confidence_interval': zswap_ci,
                'min': zswap_min
            })

        # Add parameter-specific statistics if applicable
        if param_specific_analysis:
            results['param_values'] = {}
            for value, data in param_values.items():
                results['param_values'][value] = {
                    'collection_times': {
                        'count': len(data['collection_times']),
                        'values': data['collection_times']
                    },
                    'zswap_durations': {
                        'count': len(data['zswap_durations']),
                        'values': data['zswap_durations']
                    }
                }

                # Calculate stats for this parameter value
                if data['collection_times']:
                    coll_mean, coll_total, coll_ci, coll_min = self._calculate_stats(data['collection_times'])
                    results['param_values'][value]['collection_times'].update({
                        'mean': coll_mean,
                        'total': coll_total,
                        'confidence_interval': coll_ci,
                        'min': coll_min
                    })

                if data['zswap_durations']:
                    zswap_mean, zswap_total, zswap_ci, zswap_min = self._calculate_stats(data['zswap_durations'])
                    results['param_values'][value]['zswap_durations'].update({
                        'mean': zswap_mean,
                        'total': zswap_total,
                        'confidence_interval': zswap_ci,
                        'min': zswap_min
                    })

        if verbose:
            self._print_summary(results, experiment_name, param_specific_analysis)

        return results

    def _process_run(self, run_dir, verbose=False):
        # run_name = os.path.basename(run_dir)
        result = {}

        # TODO: Ignore zswap_runtime for now, just access collection time properly
        # Collection time is located in run_*/runtime/RunTime*.out
        # run_data = self.data_import.read_parquet_dir(run_dir)
        # if 'system_info' in run_data:
        result['collection_time'] = self._extract_collection_time(run_dir)
        # if 'zswap_runtime' in run_data:
            # result['zswap_duration'] = self._extract_zswap_duration(run_data, run_name, verbose)
        return result

    # def _extract_collection_time(self, run_data, run_name, verbose=False):
    def _extract_collection_time(self, run_dir):
        full_path = ""
        for dirpath, dirnames, filenames in os.walk(run_dir):
            for filename in filenames:
                if 'RunTime' in filename:
                    full_path= os.path.join(dirpath, filename)

        sum_ms = 0
        try:
            with open(full_path, 'r') as f:
                for line in f:
                    sum_ms += int(line.strip().split(',')[2])
        except Exception as e:
            print(f"Error reading file: {str(e)}")

        sum_s = sum_ms / 1000.0
        return sum_s

    def _extract_zswap_duration(self, run_data, run_name, verbose=False):
        zswap_df = run_data['zswap_runtime']

        if "start_ts" not in zswap_df.columns or "end_ts" not in zswap_df.columns or zswap_df.is_empty():
            return None

        try:
            zswap_with_duration = zswap_df.with_columns(
                (pl.col("end_ts") - pl.col("start_ts")).alias("duration_ns")
            ).filter(pl.col("duration_ns").is_not_null())

            if zswap_with_duration.is_empty():
                return None

            total_duration_ns_run = zswap_with_duration.select(pl.sum("duration_ns")).item()
            if total_duration_ns_run is not None:
                return total_duration_ns_run / 1e9
        except Exception:
            pass

        return None

    def _calculate_stats(self, data_list):
        n = len(data_list)
        if n == 0:
            return None, None, None, None

        mean_val = np.mean(data_list)
        total_val = np.sum(data_list)
        min_val = np.min(data_list)

        ci = None
        if n > 1:
            std_dev = np.std(data_list, ddof=1)
            std_err = std_dev / np.sqrt(n)

            try:
                if std_dev == 0:
                    ci = (mean_val, mean_val)
                else:
                    t_critical = st.t.ppf(0.975, df=n-1)
                    margin_of_error = t_critical * std_err
                    ci = (mean_val - margin_of_error, mean_val + margin_of_error)
            except Exception:
                ci = None

        return mean_val, total_val, ci, min_val

    def _print_summary(self, results, experiment_name=None, param_specific_analysis=False):
        print("     --- Aggregation Summary ---")
        print(f"    Runs: {results['total_runs']}")

        # Collection Time Summary
        ct = results['collection_times']
        print(f"    --- Collection Time: {ct['count']} runs ---")

        if ct['count'] > 0:
            total = ct.get('total')
            mean = ct.get('mean')
            ci = ct.get('confidence_interval')
            min_val = ct.get('min')

            print(f"    Total: {total:.2f}s, Average: {mean:.2f}s, Min: {min_val:.2f}s")

            if ci and len(ct['values']) > 1:
                std_dev = np.std(ct['values'], ddof=1)
                print(f"    StdDev: {std_dev:.2f}s, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]s")

        # ZSwap Runtime Summary
        zs = results['zswap_durations']
        print(f"    --- ZSwap Runtime: {zs['count']} runs ---")

        if zs['count'] > 0:
            total = zs.get('total')
            mean = zs.get('mean')
            ci = zs.get('confidence_interval')
            min_val = zs.get('min')
            print(f"    Total: {total:.2f}s, Average: {mean:.2f}s, Min: {min_val:.2f}s")
            if ci and len(zs['values']) > 1:
                std_dev = np.std(zs['values'], ddof=1)
                print(f"    StdDev: {std_dev:.2f}s, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]s")

        # Parameter-specific Statistics
        if param_specific_analysis and 'param_values' in results:
            print(f"\n    --- {experiment_name} Parameter Analysis ---")
            for value, stats in results['param_values'].items():
                print(f"\n    -- {experiment_name}={value} --")
                # Collection Time
                ct = stats['collection_times']
                if ct['count'] > 0:
                    mean = ct.get('mean')
                    total = ct.get('total')
                    ci = ct.get('confidence_interval')
                    min_val = ct.get('min')
                    print(f"    Collection Time ({ct['count']} runs) - Total: {total:.2f}s, Average: {mean:.2f}s, Min: {min_val:.2f}s")
                    if ci and len(ct['values']) > 1:
                        std_dev = np.std(ct['values'], ddof=1)
                        print(f"    StdDev: {std_dev:.2f}s, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]s")

                # ZSwap Duration
                zs = stats['zswap_durations']
                if zs['count'] > 0:
                    mean = zs.get('mean')
                    total = zs.get('total')
                    ci = zs.get('confidence_interval')
                    min_val = zs.get('min')
                    print(f"    ZSwap Duration ({zs['count']} runs) - Total: {total:.2f}s, Average: {mean:.2f}s, Min: {min_val:.2f}s")
                    if ci and len(zs['values']) > 1:
                        std_dev = np.std(zs['values'], ddof=1)
                        print(f"    StdDev: {std_dev:.2f}s, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate experiment results.')
    parser.add_argument('--base-dir', default='data/curated')
    parser.add_argument('-b', '--benchmark', required=True)
    parser.add_argument('-t', '--thp-config', default='')
    parser.add_argument('-d', '--dist', default='uniform')
    parser.add_argument('-e', '--experiment', required=True)
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.base_dir)
    analyzer.analyze_experiment(
        args.benchmark,
        args.thp_config,
        args.experiment,
        dist=args.dist,
        verbose=not args.quiet
    )
