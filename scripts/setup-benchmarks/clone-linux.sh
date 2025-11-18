#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

LINUX_BENCHMARK_NAME="linux_build"
BENCHMARK_DIR_NAME="kernmlops-benchmark"

LINUX_VERSION="${LINUX_VERSION:-6.11}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"

LINUX_SOURCE_DIR="$BENCHMARK_DIR/linux_kernel"
LINUX_BUILD_DIR="$BENCHMARK_DIR/$LINUX_BENCHMARK_NAME"

if [ -d $LINUX_SOURCE_DIR ]; then
    echo "Benchmark already installed at: $LINUX_SOURCE_DIR"
    exit 0
fi

# Setup
mkdir -p $BENCHMARK_DIR
mkdir -p $LINUX_BUILD_DIR

# Install
wget "https://git.kernel.org/torvalds/t/linux-$LINUX_VERSION.tar.gz" \
    -O /tmp/linux-$LINUX_VERSION.tar.gz
tar -xzf /tmp/linux-$LINUX_VERSION.tar.gz -C $BENCHMARK_DIR
mv $BENCHMARK_DIR/linux-$LINUX_VERSION $LINUX_SOURCE_DIR

# Cleanup
rm -f /tmp/linux-$LINUX_VERSION.tar.gz
