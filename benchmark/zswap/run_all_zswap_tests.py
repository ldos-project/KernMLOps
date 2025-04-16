#!/usr/bin/env python3
import os
import subprocess
import sys
import threading
import time


def run_command(command, output_lock, command_index):
    """
    Run a command and print its output with a prefix showing which command it is.
    """
    try:
        # Create a unique log file for this command
        log_filename = f"command_{command_index}.log"

        # Start the process and redirect output to the log file
        with open(log_filename, 'w') as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Wait for the process to complete
            exit_code = process.wait()

        # Read the log file contents
        with open(log_filename, 'r') as log_file:
            output = log_file.read()

        # Print the results with the command index as a prefix
        with output_lock:
            print(f"\n[Command {command_index}]: {command}")
            print(f"[Command {command_index} Output]:\n{output}")
            print(f"[Command {command_index} Exit Code]: {exit_code}")

    except Exception as e:
        with output_lock:
            print(f"\n[Command {command_index}]: {command}")
            print(f"[Command {command_index} Error]: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <commands_file>")
        print(f"Example: {sys.argv[0]} zswap_commands.txt")
        sys.exit(1)

    commands_file = sys.argv[1]

    if not os.path.exists(commands_file):
        print(f"Error: Commands file '{commands_file}' not found.")
        sys.exit(1)

    with open(commands_file, 'r') as f:
        lines = f.readlines()

    # Filter out comments and empty lines
    commands = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            commands.append(line)

    if not commands:
        print("No commands found in the file.")
        sys.exit(1)

    print(f"Found {len(commands)} commands to execute in parallel.")

    # Create a lock for synchronizing output
    output_lock = threading.Lock()

    # Start a thread for each command
    threads = []
    for i, command in enumerate(commands):
        command_index = i+1
        print(f"Starting thread {command_index} for command: {command}...")
        thread = threading.Thread(
            target=run_command,
            args=(command, output_lock, i+1)
        )
        threads.append(thread)
        thread.start()
        # Brief pause to avoid overwhelming SSH connections
        time.sleep(0.5)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll commands have completed execution.")

if __name__ == "__main__":
    main()
