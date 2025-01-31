for ds in 8 16 32 64 128 256 512; do
    for ms in 32 64 128 256 512 1024 2048; do
        echo MAP_SIZE: ${ms} DATA_SIZE: ${ds}
        #export BENCH_CFLAGS="-DBENCH_GET_DATA_SIZE=${ds} -DBENCH_GET_ARRAY_SIZE=${ms}"
        #export BENCH_ARGS="-s${ms} -d ${ds}"
        #export KBUILD=/var/local/adityat/kbuild
        #export REPSRC=/var/local/adityat/LDOS/KernMLOps/module
        #export KHEADS=/usr/src/linux-headers-6.12.0-g0a2d0d569a5b
        #export MODULE=/
        #export SUDO="sudo"
        #make -s undeploy
        #make -s test
        #for i in $(seq 1 30); do python3 python/bench_ebpf_space.py -s ${ms} -d ${ds} 3>>ebpf-map-${ds}-${ms}.stats; done
        for i in $(seq 1 30); do ./build/test/kdev/bpf_map_bench -n 100000 -s ${ms} -d ${ds} 3>>user-mmap-${ds}-${ms}.stats; done
        #for i in `seq 1 30`; do ./build/test/e2e/bench_kernel_get/bench_kernel_get -f -n 100000 -s ${ms} -d ${ds} 3>> kmod-map-${ds}-${ms}.stats; done
        #for i in `seq 1 30`; do ./build/test/e2e/bench_kernel_get/bench_kernel_get -a -n 100000 -s ${ms} -d ${ds} 3>> kmod-array-${ds}-${ms}.stats; done
    done
done
