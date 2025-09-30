#!/bin/bash

# Get current kernel version
KVER=$(uname -r)

# Path to modules
MODULES_DIR="/lib/modules/$KVER"

# Get all available kernel module filenames (without .ko suffix)
available_modules=$(find "$MODULES_DIR" -type f -name "*.ko*" | sed -E 's|.*/||' | sed -E 's/\.ko(.xz|.gz)?$//' | sort -u)

# Get all currently loaded modules from lsmod
loaded_modules=$(lsmod | awk 'NR>1 {print $1}' | sort -u)

# Compare lists
echo "[INFO] Modules available but not currently loaded:"
comm -23 <(echo "$available_modules") <(echo "$loaded_modules")
