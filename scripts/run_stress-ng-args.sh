#!/bin/bash

BASE_PATH="/KernMLOps/scripts/"
# Path to the text file
for i in {1..10000}; do
    FILE_PATH="${BASE_PATH}stress-ng-args-${i}.txt"
    # Check if the file exists
    if [[ ! -f "$FILE_PATH" ]]; then
        echo "File not found!"
        exit 1
    fi

    # Read the file line by line
    while IFS= read -r line; do
        # Check if the line is not empty
        if [[ -n "$line" ]]; then
            # Execute the line
            eval "stress-ng $line --metrics &"
            # Sleep for 10ms
        fi
        sleep 0.01

    done < "$FILE_PATH"
done
