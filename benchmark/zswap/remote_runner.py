#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time

import paramiko


class RemoteExperimentRunner:
    def __init__(self, remote_host: str, ssh_key: str, port: int, ssh_timeout: int = 300, exp_timeout: int = 5400):
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
        # Try to connect to remote host
        try:
            if self.ssh and self.ssh.get_transport() and self.ssh.get_transport().is_active():
                return True

            logging.debug(f"Connecting to {self.remote_host}...")
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.hostname,
                username=self.username,
                port=self.port,
                key_filename=self.ssh_key,
                timeout=5
            )
            return True
        except Exception as e:
            logging.debug(f"SSH connection failed: {e}")
            return False

    def execute_command(self, command: str, ignore_errors: bool = False):
        # Execute a command on the remote host
        if not self.connect():
            if ignore_errors:
                return -1, "", "SSH connection failed"
            raise Exception("Failed to connect to remote host")

        logging.debug(f"Executing remote command: {command}")
        stdin, stdout, stderr = self.ssh.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()

        stdout_str = stdout.read().decode('utf-8')
        stderr_str = stderr.read().decode('utf-8')

        if exit_code != 0 and not ignore_errors:
            logging.error(f"Command failed (exit code {exit_code}): {command}")
            logging.error(f"STDERR: {stderr_str}")
            raise Exception(f"Remote command failed with exit code {exit_code}")

        return exit_code, stdout_str, stderr_str

    def check_ssh(self):
        try:
            # Close any existing connection to ensure we're testing a fresh connection
            if self.ssh and self.ssh.get_transport() and self.ssh.get_transport().is_active():
                self.ssh.close()
                self.ssh = None

            # Create a new SSH client for the check
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=self.hostname,
                username=self.username,
                port=self.port,
                key_filename=self.ssh_key,
                timeout=5
            )

            # Run a simple command to verify the connection works
            stdin, stdout, stderr = ssh.exec_command("echo 'SSH connection test'")
            exit_code = stdout.channel.recv_exit_status()

            # Close the connection
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

        # Ensure the SSH client is closed after reboot command
        if self.ssh:
            self.ssh.close()
            self.ssh = None

        logging.info("Waiting for machine to go down...")
        # Give the machine a moment to start shutting down
        time.sleep(5)

        # Wait until the machine is completely down (SSH fails)
        retries = 0
        max_retries = 30  # 30 seconds max to wait for shutdown
        while retries < max_retries:
            if not self.check_ssh():
                logging.info("Machine is down")
                break
            retries += 1
            time.sleep(1)

        if retries >= max_retries:
            logging.warning("Machine didn't appear to go down. Continuing anyway...")

        logging.info("Waiting for SSH to come back...")
        start_time = time.time()

        # Wait for SSH to become available again
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


    def setup_experiments(self):
        logging.info("Setting up experiment directories on remote host...")
        self.execute_command("rm -rf results && mkdir -p results")

        # Copy setup.sh script to remote host
        logging.info("Copying setup.sh to remote host...")
        local_setup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.sh")

        if os.path.exists(local_setup_path):
            rsync_cmd = [
                "rsync", "-avz",
                "-e", f"ssh -i {self.ssh_key} -p {self.port}",
                local_setup_path,
                f"{self.remote_host}:"
            ]

            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                rsync_cmd.extend(["-h", "-P", "--stats", "--progress"])

            try:
                logging.debug(f"Running command: {' '.join(rsync_cmd)}")
                subprocess.run(rsync_cmd, check=True)

                # Make sure the script is executable
                self.execute_command("chmod +x ~/setup.sh")
                logging.info("Successfully copied setup.sh to remote host")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to copy setup.sh to remote host: {e}")
                sys.exit(1)
        else:
            logging.error(f"Could not find setup.sh at {local_setup_path}")
            sys.exit(1)

    def run_experiment(self, exp_name: str, run_number: int):
        # Run a single experiment on the remote host
        result_dir = f"results/{exp_name}/run_{run_number}"
        logging.info(f"Running experiment: {exp_name}, run: {run_number}")
        logging.info(f"Result directory: {result_dir}")

        # Clean up previous experiment run and create result directory
        self.execute_command("rm -f /tmp/experiment_complete /tmp/experiment_error")
        self.execute_command(f"mkdir -p {result_dir}")

        # Start experiment in tmux
        tmux_cmd = f"""tmux new-session -d -s experiment 'bash -c "
            set -e
            (
                grep -r . /sys/module/zswap/parameters > {result_dir}/parameters.txt
                sudo bash setup.sh {result_dir}
                sudo bash -c \\\"cat /sys/fs/cgroup/benchmark_group/memory.max >> {result_dir}/parameters.txt\\\"
                sudo bash -c \\\"cat /sys/fs/cgroup/benchmark_group/memory.swap.max >> {result_dir}/parameters.txt\\\"
                touch /tmp/experiment_complete
            ) || {{
                touch /tmp/experiment_error
                exit 1
            }}
        " > {result_dir}/experiment.log 2>&1'"""

        self.execute_command(tmux_cmd)

        logging.info("Waiting for experiment to complete...")
        start_time = time.time()
        while True:
            # Check if experiment completed
            exit_code, stdout, stderr = self.execute_command(
                "[ -f /tmp/experiment_complete ] && echo 'complete' || echo 'running'",
                ignore_errors=True
            )

            if stdout.strip() == "complete":
                logging.info("Experiment completed successfully")
                # Clean up tmux session
                self.execute_command("tmux kill-session -t experiment", ignore_errors=True)
                return True

            # Check if experiment failed
            exit_code, stdout, stderr = self.execute_command(
                "[ -f /tmp/experiment_error ] && echo 'error' || echo 'running'",
                ignore_errors=True
            )

            if stdout.strip() == "error":
                logging.error("Experiment failed!!!!!!!!!!!!!!!!")
                # Clean up tmux session
                self.execute_command("tmux kill-session -t experiment", ignore_errors=True)
                return False

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.exp_timeout:
                logging.error(f"Timeout waiting for experiment to complete after {self.exp_timeout} seconds")
                self.execute_command("tmux kill-session -t experiment", ignore_errors=True)
                return False

            time.sleep(60)

    def configure_grub(self, cmdline: str):
        # configure grub with specific cmdline parameters
        if not cmdline:
            logging.error("Error: Please provide GRUB command line parameters")
            return

        logging.info(f"Configuring GRUB with parameters: {cmdline}")
        grub_config = "/etc/default/grub"

        # Escape double quotes in the cmdline for sed
        cmdline_escaped = cmdline.replace('"', '\\"')

        commands = [
            f"sudo cp -v {grub_config} {grub_config}.bak",
            f'sudo sed -i \'s/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="{cmdline_escaped}"/\' {grub_config}',
            "sudo update-grub"
        ]

        for cmd in commands:
            self.execute_command(cmd)

    def configure_cgroup_memory(self, memory_max: str, swap_max: str = None):
        # If swap_max not provided, use memory_max
        if swap_max is None:
            swap_max = memory_max

        logging.info(f"Configuring cgroup memory with memory.max: {memory_max}, memory.swap.max: {swap_max}")

        # First check if the cgroup exists
        _, stdout, _ = self.execute_command(
            "[ -d /sys/fs/cgroup/benchmark_group ] && echo 'exists' || echo 'not_exists'",
            ignore_errors=True
        )

        # Create the cgroup if it doesn't exist
        if stdout.strip() == "not_exists":
            logging.info("Creating benchmark_group cgroup as it doesn't exist")
            self.execute_command("sudo mkdir -p /sys/fs/cgroup/benchmark_group")

        # Set the memory limits
        commands = [
            f"sudo bash -c 'echo {memory_max} > /sys/fs/cgroup/benchmark_group/memory.max'",
            f"sudo bash -c 'echo {swap_max} > /sys/fs/cgroup/benchmark_group/memory.swap.max'"
        ]

        for cmd in commands:
            self.execute_command(cmd)

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

    def sync_results(self, exp_name: str, run_number: int):
        """
        Synchronize results from remote host.

        Args:
            exp_name: Name of the experiment
            run_number: Run number
        """
        remote_path = f"results/{exp_name}/run_{run_number}/"
        local_path = f"results/{exp_name}/run_{run_number}/"

        logging.info(f"Syncing results for experiment: {exp_name}, run: {run_number}")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Use rsync command line tool instead of paramiko for better performance
        rsync_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -i {self.ssh_key} -p {self.port}",
            f"{self.remote_host}:{remote_path}",
            f"{local_path}"
        ]

        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            rsync_cmd.extend(["-h", "-P", "--stats", "--progress"])

        subprocess.run(rsync_cmd, check=True)

def run_experiment_a(runner: RemoteExperimentRunner):
    print("Running Experiment A: Default configuration experiments")

    for i in range(1, 11):
        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        print(f"Running default configuration iteration {i}")
        if not runner.run_experiment("experiment_a", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_a", i)

        # Don't reboot after the last iteration
        if i < 10:
            runner.reboot_and_wait()


def run_experiment_b(runner: RemoteExperimentRunner):
    """Run Experiment B: Vary cgroup memory pressure"""
    print("Running Experiment B: Vary cgroup memory pressure")

    for mem in ["2G", "3G", "5G", "6G"]:
        print(f"Setting cgroup memory pressure to {mem}")
        for j in range(1, 6):
            # For experiment B, we set both memory.max and memory.swap.max to the same value
            # to directly observe the impact of memory pressure
            runner.configure_cgroup_memory(mem, mem)
            print(f"Running cgroup memory pressure {mem} iteration {j}")
            if not runner.run_experiment(f"experiment_b_memory_{mem}", j):
                print("Experiment failed, exiting...")
                sys.exit(1)

            runner.sync_results(f"experiment_b_memory_{mem}", j)
            runner.reboot_and_wait()


def run_experiment_c(runner: RemoteExperimentRunner):
    """Run Experiment C: Vary accept_threshold"""
    print("Running Experiment C: Vary accept_threshold")

    # for threshold in [50, 60, 70, 80, 100]:
    for threshold in [60, 70, 80, 100]:
        print(f"Setting accept_threshold to {threshold}")
        runner.configure_grub(f"zswap.enabled=1 zswap.accept_threshold_percent={threshold}")
        time.sleep(10)

        for j in range(1, 6):
            print(f"Running accept_threshold {threshold} iteration {j}")
            runner.reboot_and_wait()

            # Set default cgroup memory limits
            runner.configure_cgroup_memory("4G", "6G")
            if not runner.run_experiment(f"experiment_c_accept_threshold_{threshold}", j):
                print("Experiment failed, exiting...")
                sys.exit(1)

            runner.sync_results(f"experiment_c_accept_threshold_{threshold}", j)


def run_experiment_d(runner: RemoteExperimentRunner):
    """Run Experiment D: Vary max_pool_percent"""
    print("Running Experiment D: Vary max_pool_percent")

    for pool in [5, 10, 40, 60, 100]:
        print(f"Setting max_pool_percent to {pool}")
        runner.configure_grub(f"zswap.enabled=1 zswap.max_pool_percent={pool}")
        time.sleep(10)

        for j in range(1, 6):
            print(f"Running max_pool_percent {pool} iteration {j}")
            runner.reboot_and_wait()

            # Set default cgroup memory limits
            runner.configure_cgroup_memory("4G", "6G")
            if not runner.run_experiment(f"experiment_d_max_pool_percent_{pool}", j):
                print("Experiment failed, exiting...")
                sys.exit(1)

            runner.sync_results(f"experiment_d_max_pool_percent_{pool}", j)


def run_experiment_e(runner: RemoteExperimentRunner):
    """Run Experiment E: Vary compressor"""
    print("Running Experiment E: Vary compressor")

    compressor_modules = ["deflate", "842", "lz4", "lz4hc", "zstd"]

    for comp in compressor_modules:
        print(f"Setting compressor to {comp}")

        # Insert the required module
        runner.insert_module(comp)

        # Configure GRUB with the appropriate compressor
        runner.configure_grub(f"zswap.enabled=1 zswap.compressor={comp}")
        time.sleep(10)

        for j in range(1, 6):
            print(f"Running compressor {comp} iteration {j}")
            runner.reboot_and_wait()

            # Set default cgroup memory limits
            runner.configure_cgroup_memory("4G", "6G")
            if not runner.run_experiment(f"experiment_e_compressor_{comp}", j):
                print("Experiment failed, exiting...")
                sys.exit(1)

            runner.sync_results(f"experiment_e_compressor_{comp}", j)


def run_experiment_f(runner: RemoteExperimentRunner):
    """Run Experiment F: Vary zpool"""
    print("Running Experiment F: Vary zpool")

    for pool in ["z3fold", "zsmalloc"]:
        print(f"Setting zpool to {pool}")

        # Insert the required module
        runner.insert_module(pool)

        # Configure GRUB with the appropriate zpool
        runner.configure_grub(f"zswap.enabled=1 zswap.zpool={pool}")
        time.sleep(10)

        for j in range(1, 6):
            print(f"Running zpool {pool} iteration {j}")
            runner.reboot_and_wait()

            # Set default cgroup memory limits
            runner.configure_cgroup_memory("4G", "6G")
            if not runner.run_experiment(f"experiment_f_zpool_{pool}", j):
                print("Experiment failed, exiting...")
                sys.exit(1)

            runner.sync_results(f"experiment_f_zpool_{pool}", j)


def run_experiment_g(runner: RemoteExperimentRunner):
    """Run Experiment G: exclusive_loads ON"""
    print("Running Experiment G: exclusive_loads ON")

    runner.configure_grub("zswap.enabled=1 zswap.exclusive_loads=Y")
    time.sleep(10)

    for i in range(1, 6):
        print(f"Running exclusive_loads ON iteration {i}")
        runner.reboot_and_wait()

        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        if not runner.run_experiment("experiment_g_exclusive_loads_on", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_g_exclusive_loads_on", i)


def run_experiment_h(runner: RemoteExperimentRunner):
    """Run Experiment H: non_same_filled_pages OFF"""
    print("Running Experiment H: non_same_filled_pages OFF")

    runner.configure_grub("zswap.enabled=1 zswap.non_same_filled_pages_enabled=N")
    time.sleep(10)

    for i in range(1, 6):
        print(f"Running non_same_filled_pages OFF iteration {i}")
        runner.reboot_and_wait()

        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        if not runner.run_experiment("experiment_h_non_same_filled_pages_off", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_h_non_same_filled_pages_off", i)


def run_experiment_i(runner: RemoteExperimentRunner):
    """Run Experiment I: same_filled_pages OFF"""
    print("Running Experiment I: same_filled_pages OFF")

    runner.configure_grub("zswap.enabled=1 zswap.same_filled_pages_enabled=N")
    time.sleep(10)

    for i in range(1, 6):
        print(f"Running same_filled_pages OFF iteration {i}")
        runner.reboot_and_wait()

        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        if not runner.run_experiment("experiment_i_same_filled_pages_off", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_i_same_filled_pages_off", i)


def run_experiment_j(runner: RemoteExperimentRunner):
    """Run Experiment J: shrinker OFF"""
    print("Running Experiment J: shrinker OFF")

    runner.configure_grub("zswap.enabled=1 zswap.shrinker_enabled=N")
    time.sleep(10)

    for i in range(1, 6):
        print(f"Running shrinker OFF iteration {i}")
        runner.reboot_and_wait()

        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        if not runner.run_experiment("experiment_j_shrinker_off", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_j_shrinker_off", i)

def run_experiment_k(runner: RemoteExperimentRunner):
    """Run Experiment K: cgroup writeback OFF"""
    print("Running Experiment K: cgroup writeback OFF")

    for i in range(1, 6):
        print(f"Running cgroup writeback OFF iteration {i}")

        # Set default cgroup memory limits
        runner.configure_cgroup_memory("4G", "6G")
        # Turn off cgroup writeback
        runner.execute_command("echo 0 | sudo tee /sys/fs/cgroup/benchmark_group/memory.zswap.writeback")
        if not runner.run_experiment("experiment_k_cgroup_writeback_off", i):
            print("Experiment failed, exiting...")
            sys.exit(1)

        runner.sync_results("experiment_k_cgroup_writeback_off", i)

        # Don't reboot after the last iteration
        if i < 5:
            runner.reboot_and_wait()


def main():
    # Command line args
    parser = argparse.ArgumentParser(
        prog='remote_runner.py',
        description='Run experiments on a remote host',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="%(prog)s user@example.com C          # Run experiment C on example.com"
    )

    parser.add_argument(
        "remote_host",
        help="Remote host in format user@hostname"
    )

    parser.add_argument(
        "experiment",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        help="Experiment to run (A, B, C, D, E, F, G, H, I, J, K)"
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

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create remote runner instance
    runner = RemoteExperimentRunner(
        remote_host=args.remote_host,
        ssh_key=args.ssh_key,
        port=args.port
    )
    runner.setup_experiments()
    experiment = args.experiment.upper()

    experiment_functions = {
        "A": run_experiment_a,
        "B": run_experiment_b,
        "C": run_experiment_c,
        "D": run_experiment_d,
        "E": run_experiment_e,
        "F": run_experiment_f,
        "G": run_experiment_g,
        "H": run_experiment_h,
        "I": run_experiment_i,
        "J": run_experiment_j,
        "K": run_experiment_k
    }

    # Execute the selected experiment
    if experiment in experiment_functions:
        experiment_functions[experiment](runner)
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)

    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
