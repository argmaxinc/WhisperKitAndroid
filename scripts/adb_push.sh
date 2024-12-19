#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

for dev in `adb devices | grep -v "List" | awk '{print $1}'`
do 
  DEVICE=$dev
  break
done
if [ "$DEVICE" = "" ]; then
    echo "No Android device is connected via adb"
    exit 0
fi

echo "Pushing to: $DEVICE"

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

AXIE_TFLITE_CLI="$SOURCE_DIR/build/android/whisperax_cli"
AXIE_TFLITE_LIB="$SOURCE_DIR/build/android/libwhisperax.so"
LOCAL_LIBS="$SOURCE_DIR/external/libs/android"
LOCAL_TINY_DIR="$SOURCE_DIR/openai_whisper-tiny"
LOCAL_BASE_DIR="$SOURCE_DIR/openai_whisper-base"
LOCAL_SMALL_DIR="$SOURCE_DIR/openai_whisper-small"

REMOTE_BIN_DIR="/data/local/tmp/bin"
REMOTE_LIB_DIR="/data/local/tmp/lib"
REMOTE_SDROOT_DIR="/sdcard/argmax/tflite"
REMOTE_TINY_DIR="${REMOTE_SDROOT_DIR}/openai_whisper-tiny"
REMOTE_BASE_DIR="${REMOTE_SDROOT_DIR}/openai_whisper-base"
REMOTE_SMALL_DIR="${REMOTE_SDROOT_DIR}/openai_whisper-small"
REMOTE_INPUTS_DIR="${REMOTE_SDROOT_DIR}/inputs"

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
        if adb -s $DEVICE shell "[ ! -e $remote_path ]"; then
            adb -s $DEVICE push "$local_path" "$remote_path"
        else
            # uncomment to debug
            # echo "$remote_path already exists. Skipping push."
            true
        fi
    fi
}

# Push the files and folders to the Android device
if adb -s $DEVICE shell "[ ! -e $REMOTE_SDROOT_DIR ]"; then
    adb -s $DEVICE shell mkdir "$REMOTE_SDROOT_DIR"
fi
if adb -s $DEVICE shell "[ ! -e $REMOTE_INPUTS_DIR ]"; then
    adb -s $DEVICE shell mkdir "$REMOTE_INPUTS_DIR"
fi
if adb -s $DEVICE shell "[ ! -e $REMOTE_BIN_DIR ]"; then
    adb -s $DEVICE shell mkdir "$REMOTE_BIN_DIR"
fi
if adb -s $DEVICE shell "[ ! -e $REMOTE_LIB_DIR ]"; then
    adb -s $DEVICE shell mkdir "$REMOTE_LIB_DIR"
fi

adb -s $DEVICE push "$AXIE_TFLITE_CLI" "$REMOTE_BIN_DIR/."
adb -s $DEVICE push "$AXIE_TFLITE_LIB" "$REMOTE_LIB_DIR/."

push_if_not_exists "$LOCAL_LIBS" "$REMOTE_LIB_DIR"
push_if_not_exists "$LOCAL_TINY_DIR" "$REMOTE_TINY_DIR"
push_if_not_exists "$LOCAL_BASE_DIR" "$REMOTE_BASE_DIR"
push_if_not_exists "$LOCAL_SMALL_DIR" "$REMOTE_SMALL_DIR"
