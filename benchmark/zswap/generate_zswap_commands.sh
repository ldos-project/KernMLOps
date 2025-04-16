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
    # Break if we've processed 24 hosts
    if [ "$i" -ge 24 ]; then
        break
    fi

    HOSTNAME="${HOSTNAMES[$i]}"

    # Determine the experiment letter (cycle through EXPERIMENTS if needed)
    EXP_INDEX=$((i % NUM_EXPERIMENTS))
    EXPERIMENT="${EXPERIMENTS[$EXP_INDEX]}"

    # Determine benchmark name (add _thp suffix for hosts 13-24)
    CURRENT_BENCHMARK="$BENCHMARK_NAME"
    if [ "$i" -ge 12 ]; then
        CURRENT_BENCHMARK="${BENCHMARK_NAME}_thp"
    fi

    # Format the command
    if [ "$EXPERIMENT" = "A" ]; then
        # For experiment A, don't include the memory size parameter
        echo "./benchmark/zswap/run_zswap_test.py $HOSTNAME $EXPERIMENT $CURRENT_BENCHMARK"
    else
        # For all other experiments, include the memory size parameter
        echo "./benchmark/zswap/run_zswap_test.py -m $MEMORY_SIZE $HOSTNAME $EXPERIMENT $CURRENT_BENCHMARK"
    fi
done
