#!/bin/bash

# Set the project directory
PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

# List of build targets
TARGETS=(
    "linux"
    "qnn"
    "gpu"
)

set -eo pipefail

mkdir -p "$PROJECT_DIR/test/logs"

command -v make >/dev/null 2>&1 || { echo >&2 "make is required but not installed. Aborting."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo >&2 "cmake is required but not installed. Aborting."; exit 1; }

trap 'echo "Script interrupted. Cleaning up..."; exit 1' INT TERM ERR

for TARGET in "${TARGETS[@]}"; do
    echo "==============================="
    echo "Building target: $TARGET"
    echo "==============================="

    (cd "$PROJECT_DIR" && make build clean)

    if (cd "$PROJECT_DIR" && make build "$TARGET") > "$PROJECT_DIR/test/logs/${TARGET}_build.log" 2>&1; then
        echo "Build successful for target: $TARGET"
    else
        echo "Build failed for target: $TARGET. Check logs at $PROJECT_DIR/test/logs/${TARGET}_build.log"
        exit 1
    fi

    echo "-----------------------------------"
    echo

done

echo "All targets built successfully."

