#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

AXIE_TFLITE_CLI="$SOURCE_DIR/build/axie_tflite"
LOCAL_LIBS="$SOURCE_DIR/libs"
LOCAL_MODELS="$SOURCE_DIR/models"
LOCAL_INPUTS="$SOURCE_DIR/inputs"

REMOTE_BIN_DIR="/data/local/tmp/bin"
REMOTE_LIB_DIR="/data/local/tmp/lib"
REMOTE_MODELS_DIR="/sdcard/tflite/models"
REMOTE_INPUTS_DIR="/sdcard/tflite/inputs"

# Function to push files only if they do not exist
push_if_not_exists() {
    local local_path="$1"
    local remote_path="$2"

    if [ -d "$local_path" ]; then
        # If it's a directory, loop through its contents
        echo "Checking $remote_path ..."
        for file in "$local_path"/*; do
            local filename=$(basename "$file")
            push_if_not_exists "$file" "$remote_path/$filename"
        done
    else
        # If it's a file, check if it exists on the remote device
        if adb shell "[ ! -e $remote_path ]"; then
            adb push "$local_path" "$remote_path"
        else
            # uncomment to debug
            # echo "$remote_path already exists. Skipping push."
            true
        fi
    fi
}

# Push the files and folders to the Android device
adb push "$AXIE_TFLITE_CLI" "$REMOTE_BIN_DIR/"
push_if_not_exists "$LOCAL_LIBS" "$REMOTE_LIB_DIR"
push_if_not_exists "$LOCAL_MODELS" "$REMOTE_MODELS_DIR"
push_if_not_exists "$LOCAL_INPUTS" "$REMOTE_INPUTS_DIR"
