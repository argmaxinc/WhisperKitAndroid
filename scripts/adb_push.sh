#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

WHISPERKIT_CLI="$SOURCE_DIR/build/android/whisperkit-cli"
AXIE_TFLITE_LIB="$SOURCE_DIR/build/android/libwhisperax.so"
LOCAL_LIBS="$SOURCE_DIR/external/libs/android"
LOCAL_TINY_DIR="$SOURCE_DIR/models/openai_whisper-tiny"
LOCAL_BASE_DIR="$SOURCE_DIR/models/openai_whisper-base"
LOCAL_SMALL_DIR="$SOURCE_DIR/models/openai_whisper-small"

DEVICE_BIN_DIR="/data/local/tmp/bin"
DEVICE_LIB_DIR="/data/local/tmp/lib"
DEVICE_SDROOT_DIR="/sdcard/argmax/tflite"
DEVICE_TINY_DIR="${DEVICE_SDROOT_DIR}/models/openai_whisper-tiny"
DEVICE_BASE_DIR="${DEVICE_SDROOT_DIR}/models/openai_whisper-base"
DEVICE_SMALL_DIR="${DEVICE_SDROOT_DIR}/models/openai_whisper-small"
DEVICE_INPUTS_DIR="${DEVICE_SDROOT_DIR}/inputs"

# Function to push files only if they do not exist
push_if_not_exists() {
    local local_path="$1"
    local device_path="$2"

    if [ -d "$local_path" ]; then
        # If it's a directory, loop through its contents
        echo "Checking $device_path ..."
        for file in "$local_path"/*; do
            local filename=$(basename "$file")
            push_if_not_exists "$file" "$device_path/$filename"
        done
    else
        # If it's a file, check if it exists on the remote device
        if adb -s $DEVICE shell "[ ! -e $device_path ]"; then
            adb -s $DEVICE push "$local_path" "$device_path"
        else
            # uncomment to debug
            # echo "$device_path already exists. Skipping push."
            true
        fi
    fi
}

for dev_iter in `adb devices | grep -v "List" | awk '{print $1}'`
do 
    DEVICE=$dev_iter
    if [ "$DEVICE" = "" ]; then
        echo "No more Android device is connected via adb"
        exit 0
    fi

    echo "==================="
    echo "Pushing to: $DEVICE"

    # Push the files and folders to the Android device
    if adb -s $DEVICE shell "[ ! -e $DEVICE_SDROOT_DIR ]"; then
        adb -s $DEVICE shell mkdir "$DEVICE_SDROOT_DIR"
    fi
    if adb -s $DEVICE shell "[ ! -e $DEVICE_INPUTS_DIR ]"; then
        adb -s $DEVICE shell mkdir "$DEVICE_INPUTS_DIR"
    fi
    if adb -s $DEVICE shell "[ ! -e $DEVICE_BIN_DIR ]"; then
        adb -s $DEVICE shell mkdir "$DEVICE_BIN_DIR"
    fi
    if adb -s $DEVICE shell "[ ! -e $DEVICE_LIB_DIR ]"; then
        adb -s $DEVICE shell mkdir "$DEVICE_LIB_DIR"
    fi

    adb -s $DEVICE push "$AXIE_TFLITE_LIB" "$DEVICE_LIB_DIR/."
    adb -s $DEVICE push "$WHISPERKIT_CLI" "$DEVICE_BIN_DIR/."
    adb -s $DEVICE shell "chmod 777 $DEVICE_BIN_DIR/whisperkit-cli"
    
    push_if_not_exists "$LOCAL_LIBS" "$DEVICE_LIB_DIR"
    push_if_not_exists "$LOCAL_TINY_DIR" "$DEVICE_TINY_DIR"
    push_if_not_exists "$LOCAL_BASE_DIR" "$DEVICE_BASE_DIR"
    push_if_not_exists "$LOCAL_SMALL_DIR" "$DEVICE_SMALL_DIR"

done
