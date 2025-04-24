import logging
import sys
import time

from remote_zswap_runner import RemoteZswapRunner


class ZswapTestCases(RemoteZswapRunner):
    def __init__(
            self,
            remote_host: str,
            ssh_key: str,
            port: int,
            config: str,
            memory: str,
            single_run: bool = False,
            ssh_timeout: int = 600,
            exp_timeout: int = 5400
        ):
            super().__init__(
                remote_host=remote_host,
                ssh_key=ssh_key,
                port=port,
                config=config,
                memory=memory,
                single_run=single_run,
                ssh_timeout=ssh_timeout,
                exp_timeout=exp_timeout
            )


    # Zswap off experiment
    def zswap_off_exp(self):
        logging.info("Starting zswap off experiment...")
        self.configure_grub("zswap.enabled=0")
        time.sleep(10)
        for run in range (1, 11):
            self.reboot_and_wait()
            logging.info(f"Running experiment {run}...")
            if not self.run_experiment():
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_off", f"run_{run}_no_mem")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Zswap off with memory constraints (to induce swapping) experiment
    def zswap_off_mem_exp(self):
        logging.info("Starting zswap off with memory constraints experiment...")
        self.configure_grub("zswap.enabled=0")
        time.sleep(10)
        for run in range(1, 11):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_off_mem", f"run_{run}_mem_2_swap_8")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Zswap on experiment
    def zswap_on_exp(self):
        logging.info("Starting zswap on experiment...")
        self.configure_grub("zswap.enabled=1")
        time.sleep(10)
        for run in range (1, 11):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_on", f"run_{run}")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    def zswap_mem_tuning_exp(self):
        logging.info("Starting zswap mem tuning experiment...")
        for zswap_conf in ["0", "1"]:
            self.configure_grub(f"zswap.enabled={zswap_conf}")
            time.sleep(10)
            for mem_size in [1, 2, 4]:
                self.reboot_and_wait()
                logging.info(f"Running experiment with {mem_size}G memory")
                if not self.run_experiment(mem_size=f"{mem_size}G", swap_size=-1):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_mem_tuning", f"zswap_conf_{zswap_conf}_mem_{mem_size}_swap_8")


    # Vary zswap accept_threshold
    def zswap_accept_threshold_exp(self):
        logging.info("Starting zswap accept threshold experiment...")
        for threshold in [40, 60, 70, 80, 100]:
            self.configure_grub(f"zswap.enabled=1 zswap.accept_threshold_percent={threshold}")
            time.sleep(10)
            for run in range(1, 6):
                logging.info(f"Running experiment {run}...")
                self.reboot_and_wait()
                if not self.run_experiment(swap_size=-1):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_accept_thresh", f"thresh_{threshold}_run_{run}")
                if self.single_run:
                    logging.info("Single run mode enabled, stopping after first iteration")
                    break
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Vary max_pool_percent
    def zswap_max_pool_percent_exp(self):
        logging.info("Starting vary max pool percent experiment...")
        for pool_percent in [10, 30, 40, 50, 75]:
            self.configure_grub(f"zswap.enabled=1 zswap.max_pool_percent={pool_percent}")
            time.sleep(10)
            for run in range(1, 6):
                logging.info(f"Running experiment {run}...")
                self.reboot_and_wait()
                if not self.run_experiment(swap_size=-1):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_max_pool_pct", f"pool_{pool_percent}_run_{run}")
                if self.single_run:
                    logging.info("Single run mode enabled, stopping after first iteration")
                    break
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Vary zswap compressor
    def zswap_compressor_exp(self):
        logging.info("Starting vary compressor experiment...")
        for compressor in ["deflate", "842", "lz4", "lz4hc", "zstd"]:
            self.insert_module(compressor)
            self.configure_grub(f"zswap.enabled=1 zswap.compressor={compressor}")
            time.sleep(10)
            for run in range(1, 6):
                logging.info(f"Running experiment {run}...")
                self.reboot_and_wait()
                if not self.run_experiment(swap_size=-1):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_compressor", f"{compressor}_run_{run}")
                if self.single_run:
                    logging.info("Single run mode enabled, stopping after first iteration")
                    break
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Vary zswap zpool
    def zswap_zpool_exp(self):
        logging.info("Starting vary zpool experiment...")
        for zpool in ["z3fold", "zsmalloc"]:
            self.insert_module(zpool)
            self.configure_grub(f"zswap.enabled=1 zswap.zpool={zpool}")
            time.sleep(10)
            for run in range(1, 6):
                logging.info(f"Running experiment {run}...")
                self.reboot_and_wait()
                if not self.run_experiment(swap_size=-1):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_zpool", f"{zpool}_run_{run}")
                if self.single_run:
                    logging.info("Single run mode enabled, stopping after first iteration")
                    break
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Test with exclusive_loads enabled
    def zswap_exclusive_loads_on_exp(self):
        logging.info("Starting exclusive loads experiment...")
        self.configure_grub("zswap.enabled=1 zswap.exclusive_loads=Y")
        time.sleep(10)
        for run in range(1, 6):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_exclusive_loads_on", f"run_{run}")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Test with non_same_filled_pages on
    def zswap_non_same_filled_pages_off_exp(self):
        logging.info("Starting non same filled pages experiment...")
        self.configure_grub("zswap.enabled=1 zswap.non_same_filled_pages_enabled=Y")
        time.sleep(10)
        for run in range(1, 6):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_non_same_filled_pages_off", f"run_{run}")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Test with same_filled_pages disabled
    def zswap_same_filled_pages_off_exp(self):
        logging.info("Starting same filled pages experiment...")
        self.configure_grub("zswap.enabled=1 zswap.same_filled_pages_enabled=N")
        time.sleep(10)
        for run in range(1, 6):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_same_filled_pages_off", f"run_{run}")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Test with shrinker disabled
    def zswap_shrinker_off_exp(self):
        logging.info("Starting shrinker disabled experiment...")
        self.configure_grub("zswap.enabled=1 zswap.shrinker_enabled=N")
        time.sleep(10)
        for run in range(1, 11):
            logging.info(f"Running experiment {run}...")
            self.reboot_and_wait()
            if not self.run_experiment(swap_size=-1):
                print("Experiment failed, exiting...")
                sys.exit(1)
            self.collect(f"{self.config}_zswap_shrinker_off", f"run_{run}")
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break


    # Test with CPU cores changed
    def zswap_cpu_tuning_exp(self):
        logging.info("Starting zswap cpu tuning experiment...")
        for num_cpus in [2, 4, 8, 16, 32]:
            self.configure_grub("zswap.enabled=1")
            time.sleep(10)
            for run in range(1, 6):
                logging.info(f"Running num_cpus {num_cpus} experiment {run}...")
                self.reboot_and_wait()
                if not self.run_experiment(swap_size=-1, num_cpus=num_cpus):
                    print("Experiment failed, exiting...")
                    sys.exit(1)
                self.collect(f"{self.config}_zswap_cpu_tuning", f"zswap_cpus_{num_cpus}_run_{run}")
                if self.single_run:
                    logging.info("Single run mode enabled, stopping after first iteration")
                    break
            if self.single_run:
                logging.info("Single run mode enabled, stopping after first iteration")
                break
