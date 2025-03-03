#!/bin/bash
set -ex

USER_HOME=$(getent passwd $SUDO_USER | cut -d: -f6)

# Check if output directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi

output_dir="$1"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

cold_data="2G"

if [ "$EUID" -ne 0 ]; then
    echo "Must be run as root user"
    exit 1
fi

cat /sys/module/zswap/parameters/enabled
cat /sys/module/zswap/parameters/shrinker_enabled

mount | grep cgroup
# [ ! -d "/sys/fs/cgroup/benchmark_group" ] && mkdir -p "/sys/fs/cgroup/benchmark_group"
# echo 4G > /sys/fs/cgroup/benchmark_group/memory.max
# echo 6G > /sys/fs/cgroup/benchmark_group/memory.swap.max

sudo mkdir -p /mnt/cold_data
sudo mount -t tmpfs -o size=2G tmpfs /mnt/cold_data

# Creates 3 500MB files in cold_data
sudo dd if=/dev/urandom of=/mnt/cold_data/cold_file1 bs=1M count=500
sudo dd if=/dev/urandom of=/mnt/cold_data/cold_file2 bs=1M count=500
sudo dd if=/dev/urandom of=/mnt/cold_data/cold_file3 bs=1M count=500

# Force page cache drop to ensure these pages are cold
echo 3 >/proc/sys/vm/drop_caches

# Build Linux kernel inside cgroup
[ ! -d "$HOME/benchmark_test" ] && mkdir -p "$HOME/benchmark_test"
cd $HOME/benchmark_test
[ ! -f "linux-6.13.3.tar.xz" ] && wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.13.3.tar.xz
[ -d "linux-6.13.3" ] && rm -rf linux-6.13.3
tar xf linux-6.13.3.tar.xz
sudo apt-get update && sudo apt-get install -y \
    git \
    fakeroot \
    build-essential \
    ncurses-dev \
    xz-utils \
    libssl-dev \
    bc \
    flex \
    libelf-dev \
    bison \
    cgroup-tools

cd linux-6.13.3/
cp -v /boot/config-$(uname -r) .config
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS
make olddefconfig

sudo perf stat \
    -e 'cycles:k,instructions:k,cpu-clock,task-clock' \
    -o $HOME/$output_dir/perf_results.txt \
    sudo cgexec \
    -g memory:benchmark_group \
    make -j$(nproc) \
    &>$HOME/$output_dir/build_log.txt &&
    sudo grep -r . /sys/kernel/debug/zswap >$HOME/$output_dir/final_zswap_stats.txt
