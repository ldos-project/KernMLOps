#!/bin/bash

# Function to create destination directory if it doesn't exist
create_dest_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Main script
main() {
    # Check if correct number of arguments are provided
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <destination_dir> <source_dir1> [<source_dir2> ...]"
        exit 1
    fi

    # Set destination directory
    dest_dir="$1"
    shift

    # Create destination directory
    create_dest_dir "$dest_dir"

    # Loop through each source directory
    for src_dir in "$@"; do
        if [ ! -d "$src_dir" ]; then
            echo "Warning: $src_dir is not a directory. Skipping."
            continue
        fi

        # Use find to recursively copy files
        find "$src_dir" -type f -print0 | while IFS= read -r -d '' file; do
            # Get relative path
            rel_path="${file#$src_dir/}"
            # Create destination subdirectory if it doesn't exist
            create_dest_dir "$(dirname "$dest_dir/$rel_path")"
            # Copy file
            cp "$file" "$dest_dir/$rel_path"
        done
    done

    echo "Merge complete. All files have been copied to $dest_dir"
}

# Run the main function
main "$@"
