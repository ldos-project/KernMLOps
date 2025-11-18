#!/usr/bin/env python3
import os
import re
from fnmatch import fnmatch

import matplotlib.pyplot as plt


def parse_results(result: dict) -> dict:
    store_instr = {'experiment': result['experiment'], 'instructions': 0}

    # Open the results file
    with open(result['filename'], 'r') as f:
        for line in f:
            # Find the kernel instructions line
            if 'instructions:k' in line:
                split = line.split()
                instr = int(split[0].replace(',', ''))
                # Store into the dictionary
                store_instr['instructions'] = instr

    return store_instr

def main():
    mypath = "results/"
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(mypath) for f in filenames]
    result_files = [f for f in files if fnmatch(f, '*results.txt')]

    experiment_results = {}

    # For each file, parse the experiment name and kernel instructions
    for f in result_files:
        fname = f
        split = f.split('/')
        exp = split[1]

        # don't include any experiment_b results or writeback results
        if 'experiment_b' not in exp and 'experiment_k' not in exp:
            # replace experiment_a with default
            if exp == 'experiment_a':
                exp += '_zswap_default'

            results = {'experiment': exp, 'filename': fname}
            instr_dir = parse_results(results)

            instrs = instr_dir['instructions']

            if exp in experiment_results:
                experiment_results[exp].append(instrs)
            else:
                experiment_results[exp] = [instrs]

    # Calculate average kernel instrs for each experiment
    exp_avgs = {}
    for exp, values in experiment_results.items():
        avg = sum(values) / len(values)
        exp_avgs[exp] = {
            'runs': len(values),
            'average': f'{avg:,.0f}',
            'raw_average': avg
        }

    for exp, data in exp_avgs.items():
        print(f"Experiment: {exp}")
        print(f"  Number of runs: {data['runs']}")
        print(f"  Average instructions: {data['average']}")
        print()

    # Plot as bar graph in matplotlib
    # Create short names first
    exp_short_names = {}
    for exp in exp_avgs.keys():
        short_name = re.sub(r'experiment_[a-z]_', '', exp)
        # Change the last underscore in short_name to equals sign
        if '_' in short_name:
            last_underscore_idx = short_name.rindex('_')
            short_name = short_name[:last_underscore_idx] + '=' + short_name[last_underscore_idx+1:]
        exp_short_names[exp] = short_name

    # Find the default experiment
    default_exp = None
    for exp in exp_avgs.keys():
        if 'zswap_default' in exp:
            default_exp = exp
            break

    # Sort all other experiments by their short names alphabetically
    other_exps = [exp for exp in exp_avgs.keys() if exp != default_exp]
    sorted_other_exps = sorted(other_exps, key=lambda x: exp_short_names[x])

    # Combine with default at the front
    if default_exp:
        sorted_exps = [default_exp] + sorted_other_exps
    else:
        sorted_exps = sorted_other_exps

    # Get values in the sorted order
    short_names = [exp_short_names[exp] for exp in sorted_exps]
    avg_instrs = [exp_avgs[exp]['raw_average'] for exp in sorted_exps]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(short_names, avg_instrs)
    ax.set_ylabel('Average Kernel Instrs. per Exp. Run (trillions)', fontsize=16)
    ax.set_xlabel('Tuned zswap Parameter Experiment', fontsize=16)
    ax.set_title('Average Kernel Instructions Per Tuned zswap Parameter', fontsize=24)

    # Convert the min value to trillions to match your data
    min_value = 6_200_000_000_000  # 6 trillion (not billion)
    y_min = min_value
    y_max = max(avg_instrs) * 1.01
    ax.set_ylim(y_min, y_max)

    # Format y-axis to show trillions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e12:.2f}T'))

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('experiment_instructions_comparison.pdf')
    # plt.show()

if __name__ == '__main__':
    main()
