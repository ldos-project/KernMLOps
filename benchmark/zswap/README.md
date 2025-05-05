# Zswap Experiments

This directory contains the infrastructure for executing our zswap
performance measurements.

## Organization

The following files included are:

- `remote_zswap_runner.py`: A Python class RemoteZswapRunner containing
functions for connecting a client to a remote server, setting up
KernMLOps, configuring the machine's zswap, and collecting the results
- `run_all_zswap_exps.py`: A multi-threaded script which reads a
hostnames.txt file and executes a RemoteZswapRunner instance on each
available host in parallel for all possible Zswap configurations
- `plot_zswap_stats.py`: Parses all the results collected for each
configuration and plots the end-to-end performance for each, identifying
the optimal config, default config, and zswap off config. Includes other
helper functions for more narrow graphs as well. Relatively bespoke.

## Execution

Here are the steps taken to successfully execute these experiments:

1. Create a `hostnames.txt` file, containing a unique remote host you
   desire to run the experiment on each new line
2. From the top-level directory, run `python
   benchmark/zswap/run_all_zswap_exps.py`.

This script will run every possible Zswap configuration 5 times each on
one of the available remote hosts in `hostnames.txt`. You will observe
txt files accumulate in the `benchmark/zswap/results/` subdirectory.
Once the experiments complete, run `python plot_zswap_stats.py` to
create graphs based on the results.
