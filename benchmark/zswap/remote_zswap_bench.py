#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time

import paramiko

logging.getLogger("paramiko").setLevel(logging.CRITICAL)

class RemoteZswapRunner:
    def __init__(
        self,
        remote_host: str,
        ssh_key: str,
        port: int,
        ssh_timeout: int = 600,
        exp_timeout: int = 5400
    ):
        self.remote_host = remote_host
        self.username, self.hostname = remote_host.split('@')
        self.ssh_key = os.path.expanduser(ssh_key)
        self.port = port
        self.ssh_timeout = ssh_timeout
        self.exp_timeout = exp_timeout
        self.ssh = None
        if not os.path.exists(self.ssh_key):
            logging.error(f"SSH key not found: {self.ssh_key}")
            sys.exit(1)


    def connect(self):
        try:
            if self.ssh and self.ssh.get_transport() and self.ssh.get_transport().is_active():
                return True
            logging.info(f"Connecting to {self.remote_host}...")
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.hostname,
                username=self.username,
                port=self.port,
                key_filename=self.ssh_key,
                timeout=5
            )
            logging.info("Connected!")
            return True
        except Exception as e:
            logging.debug(f"SSH connection failed: {e}")
            return False


    def execute_command(self, command: str, ignore_errors: bool = False, get_pty: bool = False):
        if not self.connect():
            if ignore_errors:
                return -1, "", "SSH connection failed"
            raise Exception("Failed to connect to remote host")
        logging.debug(f"Executing remote command: {command}")
        stdin, stdout, stderr = self.ssh.exec_command(command, get_pty=get_pty)
        exit_code = stdout.channel.recv_exit_status()
        stdout_str = stdout.read().decode('utf-8')
        stderr_str = stderr.read().decode('utf-8')
        if exit_code != 0 and not ignore_errors:
            logging.error(f"Command failed (exit code {exit_code}): {command}")
            logging.error(f"STDERR: {stderr_str}")
            raise Exception(f"Remote command failed with exit code {exit_code}")
        else:
            logging.debug("Remote command succeeded with output:")
            logging.debug(stdout_str)
        return exit_code, stdout_str, stderr_str


    def check_ssh(self):
        try:
            # Close any existing connections
            if self.ssh and self.ssh.get_transport() and self.ssh.get_transport().is_active():
                self.ssh.close()
                self.ssh = None
            # Create new client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=self.hostname,
                username=self.username,
                port=self.port,
                key_filename=self.ssh_key,
                timeout=5
            )
            # Probe if conn works
            stdin, stdout, stderr = ssh.exec_command("echo 'SSH connection test'")
            exit_code = stdout.channel.recv_exit_status()
            ssh.close()
            return exit_code == 0
        except Exception as e:
            logging.debug(f"SSH connection check failed: {e}")
            return False


    def reboot_and_wait(self):
        logging.info("Rebooting remote machine...")
        try:
            self.execute_command("sudo reboot", ignore_errors=True)
        except Exception as e:
            logging.debug(f"Expected exception during reboot: {e}")
            pass
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        logging.info("Waiting for machine to go down...")
        time.sleep(5)
        retries = 0
        max_retries = 30
        # Wait max_retries seconds for machine to shutdown
        while retries < max_retries:
            if not self.check_ssh():
                logging.info("Machine is down")
                break
            retries += 1
            time.sleep(1)
        if retries >= max_retries:
            logging.warning("Machine didn't appear to go down. Continuing anyway...")
        # Wait for SSH to become available again
        logging.info("Waiting for SSH to come back...")
        start_time = time.time()
        while True:
            if self.check_ssh():
                logging.info("SSH is back up!")
                break
            elapsed = time.time() - start_time
            if elapsed > self.ssh_timeout:
                logging.error(f"Timeout waiting for SSH to return after {self.ssh_timeout} seconds")
                sys.exit(1)
            logging.debug(f"Still waiting for SSH... ({int(elapsed)} seconds elapsed)")
            time.sleep(5)
        logging.info("Waiting 30 more seconds for system to stabilize...")
        time.sleep(30)


    def configure_grub(self, cmdline: str):
        if not cmdline:
            logging.error("Error: Please provide GRUB command line parameters")
            return
        logging.info(f"Configuring GRUB with parameters: {cmdline}")
        grub_config = "/etc/default/grub"
        cmdline_escaped = cmdline.replace('"', '\\"')
        commands = [
            f"sudo cp -v {grub_config} {grub_config}.bak",
            f'sudo sed -i \'s/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="{cmdline_escaped}"/\' {grub_config}',
            "sudo update-grub"
        ]
        for cmd in commands:
            self.execute_command(cmd)


    def setup_experiments(self):
        # Copy local KernMLOps to remote dir
        local_kmlops_path = os.path.abspath(os.getcwd())
        logging.info(f"Copying {local_kmlops_path} to remote host...")
        if os.path.exists(local_kmlops_path):
            rsync_cmd = [
                "rsync", "-az",
                "-e", f"ssh -i {self.ssh_key} -p {self.port}",
                "--exclude=.venv",
                "--exclude=data/",
                local_kmlops_path,
                # FIXME: should be user's home directory on remote
                f"{self.remote_host}:/users/dwg/"
            ]
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                rsync_cmd.extend(["-h", "-P", "--stats", "--progress"])
            try:
                subprocess.run(rsync_cmd, check=True)
                logging.info("Successfully copied KernMLOps to remote host")
            except subprocess.CalledProcessError as e:
                logging.info(f"Failed to copy KernMLOps to remote host: {e}")
                sys.exit(1)
        else:
            logging.error(f"Could not find KernMLOps at {local_kmlops_path}")
            sys.exit(1)
        logging.info("Finished copying KernMLOps! Running setup...")
        setup_commands = [
            "sudo apt-get update && sudo apt-get install -y python3-pip",
            "cd ~/KernMLOps && pip3 install --break-system-packages -r requirements.txt",
            "echo \"export PATH=\\\"$PATH:/users/$USER/.local/bin\\\"\" >> ~/.bashrc",
            "gpg --keyserver hkp://keyserver.ubuntu.com --refresh-keys",
            "sudo apt-get install docker.io -y",
            "sudo usermod -aG docker $USER",
            "cd ~/KernMLOps && make docker-image > /dev/null 2>&1",
            # "cp -v ~/KernMLOps/config/zswap.yaml ~/KernMLOps/overrides.yaml"
        ]
        for cmd in setup_commands:
            self.execute_command(cmd)
            # Reset connection after adding user to docker group
            if "usermod" in cmd:
                self.check_ssh()


    def run_experiment(self, mem_size: int=0, swap_size: int=0, config: str="zswap_redis") -> bool:
        cmd = (
            "cd ~/KernMLOps && "
            "source ~/.bashrc && "
            "make INTERACTIVE=\"\" "
            "CONTAINER_CMD=\"bash -lc 'make install-ycsb'\" "
            "docker"
        )
        exit_code, stdout, stderr = self.execute_command(
            cmd,
            ignore_errors=True,
            get_pty=True
        )

        self.execute_command("rm -rf ~/KernMLOps/data/curated/*")
        self.execute_command(f"cp -v ~/KernMLOps/config/{config}.yaml ~/KernMLOps/overrides.yaml")
        if config == "zswap_gap":
            self.execute_command("cd ~/KernMLOps && bash scripts/setup-benchmarks/setup-gap.sh")

        logging.info(f"Running experiment with config: {config}")
        base_cmd = (
            "cd ~/KernMLOps && "
            "bash -c \'source ~/.bashrc && "
            "make INTERACTIVE=\"\" "
        )
        if mem_size < 0 or swap_size < 0:
            logging.error("Error: Memory and swap sizes must be positive integers")
            return False
        elif mem_size > 0 and swap_size > 0:
            base_cmd += f"CONTAINER_OPTS=\"--memory={mem_size}g --memory-swap={mem_size+swap_size}g\" "
        base_cmd += "collect\'"
        exit_code, stdout, stderr = self.execute_command(
            base_cmd,
            ignore_errors=True,
            get_pty=True
        )
        return exit_code == 0


    def insert_module(self, module: str):
        if not module:
            logging.error("Error: Please provide module name to insert")
            return

        logging.info(f"Inserting module into initramfs: {module}")

        sudo_cmd = (
            "sudo su - root <<'EOF'\n"
            f"echo {module} >> /etc/initramfs-tools/modules\n"
            "update-initramfs -u\n"
            "EOF"
        )

        self.execute_command(sudo_cmd)


    def collect(self, exp_name: str, run_number: str):
        local_results_path = os.path.join(os.path.dirname(f"data/curated/{exp_name}/{run_number}/"))
        if not os.path.exists(local_results_path):
            logging.debug(f"Could not find results directory at {local_results_path}, creating one...")
            os.makedirs(local_results_path)
        rsync_cmd = [
            "rsync", "-az",
            "-e", f"ssh -i {self.ssh_key} -p {self.port}",
            # FIXME: should be generic $HOME/KernMLOps/data/curated/
            f"{self.remote_host}:/users/dwg/KernMLOps/data/curated/",
            local_results_path
        ]
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            rsync_cmd.extend(["-h", "-P", "--stats", "--progress"])
        try:
            subprocess.run(rsync_cmd, check=True)
            logging.info("Successfully copied results from remote host")
        except subprocess.CalledProcessError as e:
            logging.info(f"Failed to copy results from remote host: {e}")
            sys.exit(1)


# Zswap off experiment
def zswap_off_exp(runner: RemoteZswapRunner):
    logging.info("Starting zswap off experiment...")
    runner.configure_grub("zswap.enabled=0")
    time.sleep(10)
    for i in range (1, 11):
        runner.reboot_and_wait()
        logging.info(f"Running experiment {i}...")
        if not runner.run_experiment():
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_off", f"run_{i}_no_mem")


# Zswap off with memory constraints (to induce swapping) experiment
def zswap_off_mem_exp(runner: RemoteZswapRunner):
    logging.info("Starting zswap off with memory constraints experiment...")
    runner.configure_grub("zswap.enabled=0")
    time.sleep(10)
    for i in range(1, 11):
        logging.info(f"Running experiment {i}...")
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("TEST_zswap_off_mem", f"run_{i}_mem_8_swap_8")


# Zswap on experiment
def zswap_on_exp(runner: RemoteZswapRunner):
    logging.info("Starting zswap on experiment...")
    runner.configure_grub("zswap.enabled=1")
    time.sleep(10)
    for i in range (1, 11):
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_on_quanta", f"run_{i}")


def zswap_mem_tuning_exp(runner: RemoteZswapRunner):
    logging.info("Starting zswap mem tuning experiment...")
    for zswap_conf in ["0", "1"]:
        runner.configure_grub(f"zswap.enabled={zswap_conf}")
        time.sleep(10)
        for mem_size in [4, 6, 8, 12, 16]:
            runner.reboot_and_wait()
            if not runner.run_experiment(mem_size=mem_size, swap_size=8):
                print("Experiment failed, exiting...")
                sys.exit(1)
            runner.collect("zswap_mem_tuning", f"zswap_conf_{zswap_conf}_mem_{mem_size}_swap_8")


# Vary zswap accept_threshold
def zswap_accept_threshold_exp(runner: RemoteZswapRunner):
    logging.info("Starting zswap accept threshold experiment...")
    for threshold in [40, 60, 70, 80, 100]:
        runner.configure_grub(f"zswap.enabled=1 zswap.accept_threshold_percent={threshold}")
        time.sleep(10)
        for run in range(1, 6):
            runner.reboot_and_wait()
            if not runner.run_experiment(mem_size=8, swap_size=8):
                print("Experiment failed, exiting...")
                sys.exit(1)
            runner.collect("zswap_accept_thresh", f"thresh_{threshold}_run_{run}")


# Vary max_pool_percent
def zswap_max_pool_percent_exp(runner: RemoteZswapRunner):
    logging.info("Starting vary max pool percent experiment...")
    for pool_percent in [10, 30, 40, 50, 75]:
        runner.configure_grub(f"zswap.enabled=1 zswap.max_pool_percent={pool_percent}")
        time.sleep(10)
        for run in range(1, 6):
            runner.reboot_and_wait()
            if not runner.run_experiment(mem_size=8, swap_size=8):
                print("Experiment failed, exiting...")
                sys.exit(1)
            runner.collect("zswap_max_pool_pct", f"pool_{pool_percent}_run_{run}")


# Vary zswap compressor
def zswap_compressor_exp(runner: RemoteZswapRunner):
    logging.info("Starting vary compressor experiment...")
    for compressor in ["deflate", "842", "lz4", "lz4hc", "zstd"]:
        runner.insert_module(compressor)
        runner.configure_grub(f"zswap.enabled=1 zswap.compressor={compressor}")
        time.sleep(10)
        for run in range(1, 6):
            runner.reboot_and_wait()
            if not runner.run_experiment(mem_size=8, swap_size=8):
                print("Experiment failed, exiting...")
                sys.exit(1)
            runner.collect("zswap_compressor", f"{compressor}_run_{run}")


# Vary zswap zpool
def zswap_zpool_exp(runner: RemoteZswapRunner):
    logging.info("Starting vary zpool experiment...")
    for zpool in ["z3fold", "zsmalloc"]:
        runner.insert_module(zpool)
        runner.configure_grub(f"zswap.enabled=1 zswap.zpool={zpool}")
        time.sleep(10)
        for run in range(1, 6):
            runner.reboot_and_wait()
            if not runner.run_experiment(mem_size=8, swap_size=8):
                print("Experiment failed, exiting...")
                sys.exit(1)
            runner.collect("zswap_zpool", f"{zpool}_run_{run}")


# Test with exclusive_loads enabled
def zswap_exclusive_loads_on_exp(runner: RemoteZswapRunner):
    logging.info("Starting exclusive loads experiment...")
    runner.configure_grub("zswap.enabled=1 zswap.exclusive_loads=Y")
    time.sleep(10)
    for run in range(1, 6):
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_exclusive_loads_on", f"run_{run}")


# Test with non_same_filled_pages on
def zswap_non_same_filled_pages_off_exp(runner: RemoteZswapRunner):
    logging.info("Starting non same filled pages experiment...")
    runner.configure_grub("zswap.enabled=1 zswap.non_same_filled_pages_enabled=Y")
    time.sleep(10)
    for run in range(1, 6):
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_non_same_filled_pages_off", f"run_{run}")


# Test with same_filled_pages disabled
def zswap_same_filled_pages_off_exp(runner: RemoteZswapRunner):
    logging.info("Starting same filled pages experiment...")
    runner.configure_grub("zswap.enabled=1 zswap.same_filled_pages_enabled=N")
    time.sleep(10)
    for run in range(1, 6):
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_same_filled_pages_off", f"run_{run}")


# Test with shrinker disabled
def zswap_shrinker_off_exp(runner: RemoteZswapRunner):
    logging.info("Starting shrinker disabled experiment...")
    runner.configure_grub("zswap.enabled=1 zswap.shrinker_enabled=N")
    time.sleep(10)
    for run in range(1, 11):
        logging.info(f"Running experiment {run}...")
        runner.reboot_and_wait()
        if not runner.run_experiment(mem_size=8, swap_size=8):
            print("Experiment failed, exiting...")
            sys.exit(1)
        runner.collect("zswap_shrinker_off", f"run_{run}")


def main():
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Run experiments on a remote host',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="%(prog)s user@example.com C          # Runs experiment C on example.com"
    )
    parser.add_argument(
        "remote_host",
        help="Remote host in format user@hostname"
    )
    parser.add_argument(
        "experiment",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
        help="Experiment to run (A, B, C, D, E, F, G, H, I, J, K, L)"
    )
    parser.add_argument(
        "-k", "--ssh-key",
        default="~/.ssh/cloudlab",
        help="SSH private key path (default: ~/.ssh/cloudlab)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=22,
        help="SSH port (default: 22)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    runner = RemoteZswapRunner(
        remote_host=args.remote_host,
        ssh_key=args.ssh_key,
        port=args.port,
    )
    runner.setup_experiments()
    experiment = args.experiment.upper()
    experiment_functions = {
        "A": zswap_off_exp,
        "B": zswap_off_mem_exp,
        "C": zswap_on_exp,
        "D": zswap_mem_tuning_exp,
        "E": zswap_accept_threshold_exp,
        "F": zswap_max_pool_percent_exp,
        "G": zswap_compressor_exp,
        "H": zswap_zpool_exp,
        "I": zswap_exclusive_loads_on_exp,
        "J": zswap_non_same_filled_pages_off_exp,
        "K": zswap_same_filled_pages_off_exp,
        "L": zswap_shrinker_off_exp
    }
    if experiment in experiment_functions:
        experiment_functions[experiment](runner)
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)
    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
