#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

GAP_BENCHMARK_NAME="gap"
BENCHMARK_DIR_NAME="kernmlops-benchmark"

GAP_VERSION="${GAP_VERSION:-1.5}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"

GAP_BENCHMARK_DIR="$BENCHMARK_DIR/$GAP_BENCHMARK_NAME"

if [ -d $GAP_BENCHMARK_DIR ]; then
    echo "Benchmark already installed at: $GAP_BENCHMARK_DIR"
    exit 0
fi

# Setup
mkdir -p $BENCHMARK_DIR

# Install
git clone https://github.com/sbeamer/gapbs.git \
    --branch "v$GAP_VERSION" \
    --single-branch \
    $GAP_BENCHMARK_DIR
make -C $GAP_BENCHMARK_DIR
mkdir -p "$GAP_BENCHMARK_DIR/graphs"
$GAP_BENCHMARK_DIR/converter -g 25 -b $GAP_BENCHMARK_DIR/graphs/kron25.sg
