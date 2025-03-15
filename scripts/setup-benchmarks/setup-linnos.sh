#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

LINNOS_BENCHMARK_NAME="linnos"
BENCHMARK_DIR_NAME="kernmlops-benchmark"

LAKE_BRANCH="main"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"

LINNOS_BENCHMARK_DIR="$BENCHMARK_DIR/$LINNOS_BENCHMARK_NAME"
LINNOS_DEV_0="${LINNOS_DEV_0:-/dev/nvme0n1}"
LINNOS_DEV_1="${LINNOS_DEV_1:-/dev/nvme1n1}"
LINNOS_DEV_2="${LINNOS_DEV_2:-/dev/nvme2n1}"

if [ -d $LINNOS_BENCHMARK_DIR ]; then
    echo "Benchmark already installed at: $LINNOS_BENCHMARK_DIR"
    exit 0
fi
if [ ! -b $LINNOS_DEV_0 ]; then
    echo "LinnOS device not found:  $LINNOS_DEV_0"
    exit 1
fi
if [ ! -b $LINNOS_DEV_1 ]; then
    echo "LinnOS device not found:  $LINNOS_DEV_1"
    exit 1
fi
if [ ! -b $LINNOS_DEV_2 ]; then
    echo "LinnOS device not found:  $LINNOS_DEV_2"
    exit 1
fi

# Setup
mkdir -p $BENCHMARK_DIR

# Install
git clone https://github.com/utcs-scea/LAKE.git \
    --branch "$LAKE_BRANCH" \
    --single-branch \
    $LINNOS_BENCHMARK_DIR
make -C $LINNOS_BENCHMARK_DIR/src/linnos/io_replayer
