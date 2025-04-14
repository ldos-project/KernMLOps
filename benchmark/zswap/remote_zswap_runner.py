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
        config: str,
        memory: str,
        single_run: bool = False,
        ssh_timeout: int = 600,
        exp_timeout: int = 5400
    ):
        self.remote_host = remote_host
        self.username, self.hostname = remote_host.split('@')
        self.ssh_key = os.path.expanduser(ssh_key)
        self.port = port
        self.config = config
        self.memory = memory
        self.single_run = single_run
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
                "--exclude=benchmark/zswap/pinatrace.out",
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
            "cd ~/KernMLOps && make docker-image > /dev/null 2>&1"
        ]
        for cmd in setup_commands:
            self.execute_command(cmd)
            # Reset connection after adding user to docker group
            if "usermod" in cmd:
                self.check_ssh()


    def run_experiment(self, mem_size: str="", swap_size: int=0, num_cpus: int=0) -> bool:
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
        self.execute_command(f"cp -v ~/KernMLOps/config/zswap_{self.config}.yaml ~/KernMLOps/overrides.yaml")
        if self.config == "gap":
            self.execute_command("cd ~/KernMLOps && bash scripts/setup-benchmarks/setup-gap.sh")

        logging.info(f"Running experiment with config file: zswap_{self.config}.yaml")
        base_cmd = (
            "cd ~/KernMLOps && "
            "bash -c \'source ~/.bashrc && "
            "make INTERACTIVE=\"\" "
        )
        container_opts = []

        # if mem_size parameter not given
        if not mem_size:
            # if self.memory argument was provided
            if self.memory is not None:
                container_opts.append(f"--memory={self.memory}")
                if swap_size >= -1:
                    container_opts.append(f"--memory-swap={swap_size}")
                else:
                    logging.error("Swap size must be greater than -1")
                    return False
            # else: memory not configured, so don't add param
        else: # if mem_size was given, override self.memory
            if swap_size != 0:
                container_opts.append(f"--memory={mem_size}")
                container_opts.append(f"--memory-swap={swap_size}")
            else: # swap_size == 0
                container_opts.append(f"--memory={mem_size}")

        if num_cpus < 0:
            logging.error("Error: Num CPUs must be a positive integer!")
            return False
        elif num_cpus > 0:
            container_opts.append(f"--cpus={num_cpus}")

        if container_opts:
            base_cmd += f"CONTAINER_OPTS=\"{' '.join(container_opts)}\" "

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
