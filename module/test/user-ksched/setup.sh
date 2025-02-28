#!/bin/bash
set -ex
sudo apt-get update
sudo apt-get install -y make \
    gcc \
    cmake \
    pkg-config \
    libnl-3-dev \
    libnl-route-3-dev \
    libnuma-dev \
    uuid-dev \
    libssl-dev \
    libaio-dev \
    libcunit1-dev \
    libclang-dev \
    libncurses-dev \
    meson \
    python3-pyelftools \
    g++ \
    build-essential \
    libnuma-dev \
    python3 \
    python3-pip \
    python3-pyelftools \
    pkg-config \
    meson \
    ninja-build \
    libaio-dev \
    libcunit1-dev \
    uuid-dev \
    libjson-c-dev \
    libssl-dev \
    libncurses5-dev \
    libncursesw5-dev

# Rollback and patch Caladan
# clone if doesn't exist
if [ ! -d "caladan" ]; then
    git clone https://github.com/shenango/caladan.git
    cd caladan
    git reset --hard 14a57f0f405cdbf54f897436002ee472ede2ca40
    git apply ../caladan.patch
else
    cd caladan
fi

# Build and insert ksched module
make submodules
export LIBRARY_PATH=$LIBRARY_PATH:$(dirname $(gcc -print-libgcc-file-name))
make clean && make
pushd ksched
make clean && make
mv *.ko build
popd
sudo ./scripts/setup_machine.sh && lsmod | grep ksched

# Compile and test the benchmark script
cd ..
g++ -Wall -Werror -O3 -o user_ksched user_ksched.cpp
sudo bash -c './user_ksched -n 100000 -s 32 -d 8 3>> ksched-user-8-32.stats 4>> returner'
