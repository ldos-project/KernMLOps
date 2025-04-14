#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <hostnames_file> <memory_size_in_GB> <benchmark_name>"
    echo "Example: $0 hostnames.txt 2 redis"
    exit 1
fi

HOSTNAMES_FILE="$1"
MEMORY_SIZE="${2}G"
BENCHMARK_NAME="$3"

# Check if the hostnames file exists
if [ ! -f "$HOSTNAMES_FILE" ]; then
    echo "Error: Hostnames file '$HOSTNAMES_FILE' not found."
    exit 1
fi

# Read hostnames into an array (macOS compatible)
# Skip empty lines and lines starting with #
HOSTNAMES=()
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and commented lines
    if [ -n "$line" ] && [[ ! "$line" =~ ^[[:space:]]*# ]]; then
        HOSTNAMES+=("$line")
    fi
done <"$HOSTNAMES_FILE"

# Define experiment letters (excluding D)
EXPERIMENTS=(A B C E F G H I J K L M)
NUM_EXPERIMENTS=${#EXPERIMENTS[@]}

# Output header
echo "# ZSwap Test Commands"
echo "# Memory Size: $MEMORY_SIZE"
echo "# Benchmark: $BENCHMARK_NAME"
echo ""

# Generate commands
for i in "${!HOSTNAMES[@]}"; do
    # Make sure we don't exceed the number of experiments
    if [ "$i" -ge "$NUM_EXPERIMENTS" ]; then
        echo "# Warning: More hostnames than experiments. Stopping at experiment ${EXPERIMENTS[$NUM_EXPERIMENTS - 1]}."
        break
    fi

    HOSTNAME="${HOSTNAMES[$i]}"
    EXPERIMENT="${EXPERIMENTS[$i]}"

    if [ "$EXPERIMENT" = "A" ]; then
        # For experiment A, don't include the memory size parameter
        echo "./benchmark/zswap/run_zswap_test.py $HOSTNAME $EXPERIMENT $BENCHMARK_NAME"
    else
        # For all other experiments, include the memory size parameter
        echo "./benchmark/zswap/run_zswap_test.py -m $MEMORY_SIZE $HOSTNAME $EXPERIMENT $BENCHMARK_NAME"
    fi
done
