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
import copy
import queue
import signal
import sys
import threading
import time

from remote_zswap_runner import RemoteZswapRunner

ZSWAP_CONFIGS = [
    'workloads', 'compressor', 'zpool', 'max_pool_percent',
    'accept_threshold_percent', 'shrinker_enabled', 'exclusive_loads',
    'same_filled_pages_enabled', 'non_same_filled_pages_enabled'
]

shutdown_event = threading.Event()
def signal_handler(sig, frame):
    print("\nReceived SIGTERM. Initiating graceful shutdown...")
    shutdown_event.set()

def read_hostnames_from_file(filename: str):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if (line.strip() and line[0] != '#')]
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
    def start_zswap_experiment(self, config: dict[str, str]):
        # Block thread while available_hosts is empty
        host_id = None
        while not shutdown_event.is_set():
            try:
                host_id = self.available_hosts.get(block=True, timeout=1)
                break
            except queue.Empty:
                continue
        # If shutdown was requested while waiting for a host, return immediately
        if shutdown_event.is_set():
            if host_id is not None:
                self.available_hosts.put(host_id)
            return
        # Actually trigger the experiment on the remote host
        try:
            runner = RemoteZswapRunner(
                remote_host=host_id,
                ssh_key='~/.ssh/cloudlab'
            )
            runner.establish_connection()
            benchmark = config.pop('workloads')
            runner.configure_zswap(parameter='enabled', value='Y')
            runner.config_zswap_params(config)
            runner.setup_kernmlops(owner='dariusgrassi', branch='zswap-runner')
            runner.setup_ycsb_experiment(benchmark=benchmark)
            runner.shrink_page_cache()
            runner.clear_swap()
            print('Setup complete, running benchmark...')
            log_fname = benchmark + '_' + '_'.join(config.values())
            runner.run_mem_constrained_ycsb_experiment(benchmark=log_fname)
            runner.find_and_parse_logfiles(log_fname + '*')
        # acquire lock to queue and return host to pool when done
        finally:
            with self.thread_lock:
                if host_id is not None:  # Make sure we have a valid host_id
                    self.available_hosts.put(host_id)
                    print(f"Done. Released host {host_id} back to the pool!")

def run_all_experiments():
    hosts = read_hostnames_from_file('hostnames.txt')
    pool = RemoteHostThreadPool(hosts)
    config_queue = queue.Queue()
    for workloads in ['redis', 'mongodb']:
        for compressor in ['lzo', 'deflate', '842', 'lz4', 'lz4hc', 'zstd']:
            for zpool in ['zbud', 'z3fold', 'zsmalloc']:
                for max_pool_percent in ['10', '20', '40']:
                    for accept_threshold_percent in ['80', '90', '100']:
                        for shrinker_enabled in ['Y','N']:
                            for exclusive_loads in ['Y','N']:
                                for same_filled_pages_enabled in ['Y','N']:
                                    for non_same_filled_pages_enabled in ['Y','N']:
                                        config = {name: value
                                            for name,value in locals().items()
                                            if name in ZSWAP_CONFIGS}
                                        config_queue.put(config)
    workers = []
    for _ in range(len(hosts)):
        worker = threading.Thread(
            target=worker_function,
            args=(pool, config_queue),
            daemon=True
        )
        workers.append(worker)
        worker.start()
    try:
        # Wait for all tasks to be processed or for shutdown signal
        while not config_queue.empty() and not shutdown_event.is_set():
            time.sleep(1)
        # If shutdown was requested, wait for workers to finish current tasks
        if shutdown_event.is_set():
            print("Waiting for workers to finish current tasks...")
            timeout = 30  # Give workers 30 seconds to finish
            for worker in workers:
                worker.join(timeout=timeout / len(workers))
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Initiating shutdown...")
        shutdown_event.set()

def worker_function(pool, config_queue):
    while not shutdown_event.is_set():
        try:
            config = config_queue.get(block=False)
            try:
                for i in range(1, 6):
                    # Avoid dict being passed by reference
                    config_copy = copy.deepcopy(config)
                    pool.start_zswap_experiment(config_copy)
            finally:
                config_queue.task_done()
        except queue.Empty:
            time.sleep(1)

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    run_all_experiments()
    if shutdown_event.is_set():
        print("Shutdown complete.")
    else:
        print("All experiments completed successfully.")
