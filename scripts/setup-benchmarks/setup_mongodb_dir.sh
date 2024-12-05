#!/bin/bash
# Define variables first
YCSB_BENCHMARK_NAME="ycsb"
BENCHMARK_DIR_NAME="kernmlops-benchmark"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"
YCSB_BENCHMARK_DIR="$BENCHMARK_DIR/$YCSB_BENCHMARK_NAME"

# Then use them
if [ -d "$YCSB_BENCHMARK_DIR/mongo_db" ]; then
    echo "Directory $YCSB_BENCHMARK_DIR/mongo_db already exists."
    exit 0
fi
mkdir -p "$YCSB_BENCHMARK_DIR/mongo_db"
