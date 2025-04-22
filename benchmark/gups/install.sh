#!/bin/bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    libopenblas-dev
git clone https://github.com/technion-csl/gups.git
make -C gups
# run gups with a 1gb sized array
cd gups && ./gups --log2_length=27
