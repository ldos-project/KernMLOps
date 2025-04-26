"""
# TODO: run tests with zswap off, no swapping
# TODO: run tests with zswap off, swapping
- workloads: [redis, mongodb]
- compressor: [lzo, deflate, 842, lz4, lz4hc, zstd]
- zpool: [zbud, z3fold, zsmalloc]
- max_pool_percent: [10, 20, 40]
- accept_threshold_percent: [80, 90, 100]
- shrinker_enabled: Y/N
- exclusive_loads: Y/N
- same_filled_pages_enabled: Y/N
- non_same_filled_pages_enabled: Y/N
2 * 6 * 3 * 3 * 3 * 2 * 2 * 2 * 2 configurations = 5,184
5 runs per configuration = 25,920 runs total
Maximum 40 runs in parallel = 648 synchronous runs
10 minutes per run = 6,480 minutes = 108 hrs = 4.5 days
"""
import queue
import sys
import threading
import time

from remote_zswap_runner import RemoteZswapRunner


def read_hostnames_from_file(filename: str):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

class RemoteHostThreadPool:
    def __init__(self, host_ids):
        self.max_hosts = len(host_ids)
        print(f"Max hosts: {self.max_hosts}")
        # for indicating to new threads which hosts are available
        self.available_hosts = queue.Queue()
        for host_id in host_ids:
            self.available_hosts.put(host_id)
        self.running_threads = []
        self.thread_lock = threading.Lock()

    # A single thread instantiates RemoteZswapRunner
    # XXX: Doesn't start an experiment for now, just establishes connection
    def start_zswap_experiment(self, config):
        # Block thread while available_hosts is empty
        host_id = None
        while True:
            try:
                host_id = self.available_hosts.get(block=True, timeout=1)
                break
            except queue.Empty:
                continue
        try:
            print(f"Starting experiment on host {host_id}")
            runner = RemoteZswapRunner(
                remote_host=host_id,
                ssh_key='~/.ssh/cloudlab'
            )
            runner.establish_connection()
            time.sleep(10)
        # acquire lock to queue and return host to pool when done
        finally:
            with self.thread_lock:
                self.available_hosts.put(host_id)
                print(f"Done. Released host {host_id} back to the pool!")

    # Spawn thread for zswap experiment
    def spawn_thread(self, config):
        t = threading.Thread(target=self.start_zswap_experiment, args=(config,), daemon=True)
        with self.thread_lock:
            self.running_threads.append(t)
        t.start()
        return t

    def spawn_threads(self, configs):
        threads = []
        for config in configs:
            thread = self.spawn_thread(config)
            threads.append(thread)
        return threads

    def wait_for_all_threads(self):
        for thread in self.running_threads:
            thread.join()

    def shutdown(self):
        self.should_terminate = True
        self.wait_for_all_threads()

if __name__ == "__main__":
    hosts = read_hostnames_from_file('hostnames.txt')
    pool = RemoteHostThreadPool(hosts)
    for i in range(1,81):
        config = f"run_{i}"
        print(config)
        t = pool.spawn_thread(config)
    pool.wait_for_all_threads()
