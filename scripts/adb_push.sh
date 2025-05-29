#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This script is used to push the files and folders to all connected Android devices
#
# whisperkit-cli: CLI app to run transcription on Android device, with files input.
# libwhisperkit.so: main whisperkit lib
# models: all models for whisperkit in models/ folder
# test/jfk_441khz.m4a: sample test audio file for whisperkit
# scripts/run_on_android.sh: script to run the app on Android device

# Usage: make adb-push [forced]
#        or ./adb_push.sh [forced]
# If forced is provided, the script will push the files and folders to 
#   the Android device even if they already exist. Otherwise, it will
#   skip the push if the files already exist.

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

WHISPERKIT_CLI="$SOURCE_DIR/build/android/whisperkit-cli"
AXIE_TFLITE_LIB="$SOURCE_DIR/build/android/libwhisperkit.so"
LOCAL_LIBS="$SOURCE_DIR/external/libs/android"
LOCAL_MODELS_DIR="$SOURCE_DIR/models"

DEVICE_BIN_DIR="/data/local/tmp/bin"
DEVICE_LIB_DIR="/data/local/tmp/lib"
DEVICE_SDROOT_DIR="/sdcard/argmax/tflite"
DEVICE_MODELS_DIR="${DEVICE_SDROOT_DIR}/models"
DEVICE_INPUTS_DIR="${DEVICE_SDROOT_DIR}/inputs"

EXEC_SCRIPT="$SOURCE_DIR/scripts/run_on_android.sh"
TEST_AUDIO_FILE="$SOURCE_DIR/test/jfk_441khz.m4a"

FORCED=$1
if  [ "$FORCED" = "forced" ]; then
    echo "adb push in forced mode.."
fi

# Function to push files only if they do not exist
push_if_not_exists() {
    local local_path="$1"
    local device_path="$2"
    local forced="$3"

    if [ -d "$local_path" ]; then
        # If it's a directory, loop through its contents
        echo "Checking $device_path ..."
        for file in "$local_path"/*; do
            local filename=$(basename "$file")
            push_if_not_exists "$file" "$device_path/$filename" $forced
        done
    else
        # If it's a file, check if it exists on the remote device
        if  [ "$forced" = "forced" ]; then
            adb -s $DEVICE push "$local_path" "$device_path"
        elif adb -s $DEVICE shell "[ ! -e $device_path ]"; then
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

    push_if_not_exists "$AXIE_TFLITE_LIB" "$DEVICE_LIB_DIR/." forced
    push_if_not_exists "$WHISPERKIT_CLI" "$DEVICE_BIN_DIR/." forced
    push_if_not_exists "$LOCAL_LIBS" "$DEVICE_LIB_DIR" $FORCED
    adb -s $DEVICE push "$EXEC_SCRIPT" "$DEVICE_SDROOT_DIR"
    adb -s $DEVICE push "$TEST_AUDIO_FILE" "$DEVICE_SDROOT_DIR/inputs/"
    adb -s $DEVICE shell "chmod 777 $DEVICE_SDROOT_DIR/run_on_android.sh"
    adb -s $DEVICE shell "chmod 777 $DEVICE_BIN_DIR/whisperkit-cli"
   
    push_if_not_exists "$LOCAL_MODELS_DIR" "$DEVICE_MODELS_DIR" $FORCED

done
