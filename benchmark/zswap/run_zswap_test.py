#!/usr/bin/env python3
import argparse
import logging
import sys

from zswap_tests import ZswapTestCases


def main():
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Run experiments on a remote host',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="%(prog)s user@example.com C          # Runs experiment C on example.com"
    )
    parser.add_argument(
        "remote_host",
        help="Remote host in format user@hostname"
    )
    parser.add_argument(
        "experiment",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
        help="Experiment to run (A, B, C, D, E, F, G, H, I, J, K, L, M)"
    )
    parser.add_argument(
        "config",
        choices=[
            "redis_uniform", "redis_uniform_thp",
            "redis_zipfian", "redis_zipfian_thp",
            "mongodb_uniform", "mongodb_uniform_thp",
            "mongodb_zipfian", "mongodb_zipfian_thp",
            "gap", "gap_thp"
        ],
        help="Configuration file for benchmark experiment"
    )
    parser.add_argument(
        "-m", "--memory",
        default=None,
        help="Memory restrictions for benchmark container"
    )
    parser.add_argument(
        "-k", "--ssh-key",
        default="~/.ssh/cloudlab",
        help="SSH private key path (default: ~/.ssh/cloudlab)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=22,
        help="SSH port (default: 22)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-s", "--single-run",
        action="store_true",
        help="Run experiment only once (override default iterations)"
    )
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    runner = ZswapTestCases(
        remote_host=args.remote_host,
        ssh_key=args.ssh_key,
        port=args.port,
        config=args.config,
        memory=args.memory,
        single_run=args.single_run
    )
    runner.setup_experiments()
    experiment = args.experiment.upper()
    experiment_functions = {
        "A": runner.zswap_off_exp,
        "B": runner.zswap_off_mem_exp,
        "C": runner.zswap_on_exp,
        "D": runner.zswap_mem_tuning_exp,
        "E": runner.zswap_accept_threshold_exp,
        "F": runner.zswap_max_pool_percent_exp,
        "G": runner.zswap_compressor_exp,
        "H": runner.zswap_zpool_exp,
        "I": runner.zswap_exclusive_loads_on_exp,
        "J": runner.zswap_non_same_filled_pages_off_exp,
        "K": runner.zswap_same_filled_pages_off_exp,
        "L": runner.zswap_shrinker_off_exp,
        "M": runner.zswap_cpu_tuning_exp
    }
    if experiment in experiment_functions:
        experiment_functions[experiment]()
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)
    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
