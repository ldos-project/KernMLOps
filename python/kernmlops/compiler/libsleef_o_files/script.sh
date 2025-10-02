#!/bin/bash

archive="libsleef.a"
output_dir="extracted_files"

# Check if the archive exists
if [[ ! -f "$archive" ]]; then
    echo "Error: Archive '$archive' not found."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

echo "Extracting all files from '$archive' into '$output_dir'..."

# Get a list of all files in the archive, in their internal order.
files=$(ar -t "$archive")

# An associative array to keep track of the instance count for each filename
declare -A counts

# Loop through each file name listed by 'ar -t'
for file in $files; do
    # Get the current count for this file, defaulting to 0
    current_count=${counts["$file"]:-0}

    # Increment the count for the next instance
    next_count=$((current_count + 1))
    counts["$file"]=$next_count

    echo "Extracting '$file' (instance $next_count)"

    # Use 'ar -xN' with the count to extract the specific instance.
    # The file is extracted to the current directory with its original name.
    ar -xN "$next_count" "$archive" "$file"

    # Create a unique name for the extracted file
    if [[ $next_count -gt 1 ]]; then
        unique_name="${file%.*}_${next_count}.${file##*.}"
    else
        unique_name="$file"
    fi

    # Move the extracted file to the output directory and give it the unique name
    mv "$file" "$output_dir/$unique_name"
done

echo "Extraction complete."
