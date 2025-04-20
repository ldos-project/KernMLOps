import os
import subprocess
import sys


def launch_subprocess(script_path):
    # Copy the current environment variables
    env = os.environ.copy()

    # Update PYTHONPATH to include all directories from the parent's sys.path
    env['PYTHONPATH'] = os.pathsep.join(sys.path)

    # Launch the subprocess using the same Python interpreter
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Subprocess output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed with error:\n", e.stderr)

if __name__ == "__main__":
    # Replace 'child_script.py' with the path to your subprocess script
    launch_subprocess("child_script.py")
