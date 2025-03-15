#!/bin/bash

# Define variables first
YCSB_BENCHMARK_NAME="ycsb"
BENCHMARK_DIR_NAME="kernmlops-benchmark"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"
YCSB_BENCHMARK_DIR="$BENCHMARK_DIR/$YCSB_BENCHMARK_NAME"

echo "Setting up Redis benchmark..."

# Install Redis server if not already installed
if ! command -v redis-server &>/dev/null; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            sudo apt-get update
            sudo apt-get install -y redis-server
        elif [ -f /etc/redhat-release ]; then
            sudo dnf install -y redis
        fi
    else
        echo "Unsupported operating system"
        exit 1
    fi
fi

# Create Redis data directory
REDIS_DATA_DIR="${BENCHMARK_DIR}/redis"
if [ -d "$REDIS_DATA_DIR" ]; then
    echo "Directory $REDIS_DATA_DIR already exists."
else
    mkdir -p "$REDIS_DATA_DIR"
fi

# Create Redis configuration
cat >"${REDIS_DATA_DIR}/redis.conf" <<EOF
port 6379
dir ${REDIS_DATA_DIR}
maxmemory 82gb
maxmemory-policy allkeys-lru
EOF

# Copy YCSB workload configuration for Redis
YCSB_WORKLOAD_DIR="${YCSB_BENCHMARK_DIR}/ycsb-0.17.0/workloads"
mkdir -p "${YCSB_WORKLOAD_DIR}"
cp "scripts/setup-benchmarks/redis-workload.properties" "${YCSB_WORKLOAD_DIR}/workloada-redis"

echo "Redis benchmark setup complete"
