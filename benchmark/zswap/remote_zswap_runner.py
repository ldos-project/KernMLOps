import os
import sys
import time

import paramiko

DEFAULT_MEM_SIZE='1G' # the default memory constraint for experiments

class RemoteZswapRunner:
    def __init__(self, remote_host, ssh_key, port, ssh_timeout=600):
        self.remote_host = remote_host
        self.username, self.hostname = remote_host.split('@')
        self.ssh = None
        self.ssh_key = os.path.expanduser(ssh_key)
        if not os.path.exists(self.ssh_key):
            print(f"SSH key not found (Path: {self.ssh_key})")
            sys.exit(1)
        self.port = port
        self.ssh_timeout = ssh_timeout

    def establish_connection(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        try:
            self.ssh.connect(
                hostname=self.hostname,
                username=self.username,
                port=self.port,
                key_filename=self.ssh_key,
                timeout=5)
            print(f"Successfully connected to {self.hostname}")
        except Exception:
            self.ssh = None
        return self.ssh

    def reset_connection(self):
        print('Resetting SSH connection...')
        if self.ssh and self.ssh.get_transport() and self.ssh.get_transport().is_active():
            self.ssh.close()
        return self.establish_connection()

    def execute_remote_command(self, command: str, get_pty: bool=False, verbose: bool=False):
        if self.ssh:
            print(f"Executing remote command: {command}")
            stdin, stdout, stderr = self.ssh.exec_command(command, get_pty=get_pty)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code == 0:
                if verbose:
                    stdout_str = stdout.read().decode('utf-8').strip()
                    print('Remote command output:')
                    print(stdout_str)
                return 0
            else:
                stderr_str = stderr.read().decode('utf-8')
                print('Command failed!')
                print(f"STDERR: {stderr_str}")
        return -1

    def reboot_machine(self):
        print('Attempting to reboot machine...')
        self.execute_remote_command('sudo shutdown -r now')
        if self.ssh:
            self.ssh = None
        retry_timer = 0
        print('Waiting for machine to come back up...')
        while retry_timer < self.ssh_timeout:
            time.sleep(5)
            if not self.establish_connection():
                retry_timer += 5
            else:
                break
        if retry_timer >= self.ssh_timeout:
            print('SSH timeout expired!')
            return -1
        if self.reset_connection():
            print('SSH is back up! Waiting for startup to finish...')
        time.sleep(30)
        return 0

    # configures zswap parameters through the sysfs interface
    def configure_zswap(self, parameter: str, value: str):
        param_options = [
            'accept_threshold_percent',
            'enabled',
            'max_pool_percent',
            'same_filled_pages_enabled',
            'zpool',
            'compressor',
            'exclusive_loads',
            'non_same_filled_pages_enabled',
            'shrinker_enabled'
        ]
        if parameter not in param_options:
            print(f"ERROR: zswap param \'{parameter}\' not available")
            return -1
        zswap_path = '/sys/module/zswap/parameters/'
        zswap_param_path = zswap_path + parameter
        zswap_command = f"echo {value} | sudo tee {zswap_param_path}"
        self.execute_remote_command(zswap_command)
        return 0

    """
    def setup_kernmlops(self, verbose=False):
        self.execute_remote_command('git clone https://github.com/ldos-project/KernMLOps.git', verbose=verbose)
        # skip building the docker image and activating the virtual environment
        self.execute_remote_command('cd KernMLOps/ && sed -i \'/^make_docker_image$/s/^/# /\' scripts/setup_prep_env.sh', verbose=verbose)
        self.execute_remote_command('cd KernMLOps/ && sed -i \'/^source_shell$/s/^/# /\' scripts/setup_prep_env.sh', verbose=verbose)
        # install uv and docker, setup venv
        self.execute_remote_command('cd KernMLOps/ && source scripts/setup_prep_env.sh', verbose=verbose)
        # TODO: pull kmlops docker image from dockerhub
        return 0

    # should run a bunch of sysctl commands to aggressively shrink the
    # Linux page cache so it doesn't interfere with our zswap
    # measurements we do this because the page cache is system-wide,
    # whereas it would be full if the entire machine's memory was
    # *actually* under enough pressure to invoke zswap
    def shrink_page_cache(self):

    # provides option to setup either redis or mongodb benchmark
    # should run make install-yscb and copy over correct config
    def setup__ycsb_experiment(self, benchmark: str):
        - pull kernmlops repository (pull my local repo branch for now)
        - pull up-to-date docker image from dockerhub to avoid lengthy build times

    # should use gups install script to setup gups benchmark tool
    def setup_gups_experiment(self):
        pass

    # this should take a parameter to constrain the memory by some amount
    # this should run make collect and pipe the output to a log file
    def run_mem_constrained_kmlops_benchmark(self, mem=DEFAULT_MEM_SIZE):
        pass

    # collect all run logs
    # this should use rsync to collect all the benchmark log files
    def collect_logs(self):
        pass

    # this should walk through a single ycsb benchmark log file and aggregate the runtimes
    def parse_ycsb_logfile_runtime(self):
        pass

    # this should either create or append to a results file
    # used for saving the results of all the benchmarks
    def save_resultsfile(self, filename: str, text: str):
        pass

    # this should walk through all the logfiles for that run and calculate the mean, std, and 95% ci
    def parse_all_benchmark_runtimes(self):
        pass
    """

# For testing RemoteZswapRunner functionality
def main():
    print("Testing out the RemoteZswapRunner class...")
    runner = RemoteZswapRunner(
        remote_host='dwg@pc701.emulab.net',
        ssh_key='~/.ssh/cloudlab',
        port=22)
    runner.establish_connection()
    runner.reset_connection()
    # runner.reboot_machine()
    runner.configure_zswap(parameter='enabled', value='1')
    # runner.setup_kernmlops(verbose=True)

if __name__ == '__main__':
    main()
