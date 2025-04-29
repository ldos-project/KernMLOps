#!/bin/bash
set -ex
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cgroup-tools
# setup cgroup
CGROUP_NAME="gups"
sudo cgcreate -g memory:$CGROUP_NAME
sudo cgset -r memory.max='1G' $CGROUP_NAME
sudo cgget -r memory.max $CGROUP_NAME
# turn off the page cache
sudo sysctl -w vm.vfs_cache_pressure=500       # evict filesystem cache data
sudo sysctl -w vm.dirty_background_bytes=16384 # minimize dirty pages in memory
sudo sysctl -w vm.dirty_bytes=32768
sudo sysctl -w vm.dirty_expire_centisecs=100 # flush dirty data to disk quickly
sudo sysctl -w vm.dirty_writeback_centisecs=100
# optionally, enable zswap
if [[ "$*" == *"-z"* ]]; then
    echo 1 | sudo tee /sys/module/zswap/parameters/enabled
    echo "zswap enabled"
else
    echo 0 | sudo tee /sys/module/zswap/parameters/enabled
    echo "zswap disabled"
fi
# drop caches
free && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches && free
# install gups
if [ ! -d "gups" ]; then
    git clone https://github.com/technion-csl/gups.git
fi
make -C gups
# run benchmark inside cgroup
cd gups && sudo cgexec -g memory:$CGROUP_NAME ./gups --log2_length=27
