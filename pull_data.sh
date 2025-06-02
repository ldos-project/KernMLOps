#!/bin/bash
REMOTE_DIR="/users/dwg/KernMLOps/data/curated"
LOCAL_DIR="./curated_data"
mkdir -p "$LOCAL_DIR"

MAX_CONCURRENT=15

while read -r HOST; do
    # Count currently running processes
    RUNNING=$(jobs -p | wc -l)

    # Wait if we've reached the maximum
    while [ $RUNNING -ge $MAX_CONCURRENT ]; do
        sleep 1
        RUNNING=$(jobs -p | wc -l)
    done

    {
        echo "Cleaning old files and pulling data from $HOST..."
        mkdir -p "$LOCAL_DIR/${HOST##*@}"

        # Delete old files before May 7 00:00 on remote host
        ssh "$HOST" 'find /users/dwg/KernMLOps/data/curated/redis -type f ! -newermt "2025-05-07 00:00:00" -delete'

        rsync -azP "$HOST:$REMOTE_DIR/redis/" "$LOCAL_DIR/${HOST##*@}/redis/"

        echo "Transfer from $HOST complete."
    } &

done <hostnames.txt

wait
echo "All transfers complete."
