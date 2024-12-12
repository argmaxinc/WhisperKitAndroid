#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

REMOTE_SDROOT_DIR="/sdcard/argmax/tflite"
REMOTE_INPUTS_DIR="${REMOTE_SDROOT_DIR}/inputs"
REMOTE_BIN_DIR="/data/local/tmp/bin"
REMOTE_LIB_DIR="/data/local/tmp/lib"

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
LINUX_BUILD_DIR=./build_linux
ARG=$1

case $ARG in
    "linux")
        echo "  ${0} linux   : run in Docker"
        cd $SOURCE_DIR
        $LINUX_BUILD_DIR/whisperax_cli test/jfk_441khz.m4a tiny
        exit 0 ;;

    "gpu" | "qnn" | "" )
        echo "  ${0} [gpu|qnn] : run on Host PC"

        for dev in `adb devices | grep -v "List" | awk '{print $1}'`
        do 
            DEVICE=$dev
            break
        done
        if [ "$DEVICE" = "" ]; then
            echo "No Android device is connected via adb"
            exit -1
        fi
        echo "Test on: $DEVICE"

        CMD="cd ${REMOTE_SDROOT_DIR} && \
            export LD_LIBRARY_PATH=${REMOTE_LIB_DIR} && \
            ${REMOTE_BIN_DIR}/whisperax_cli \
            ${REMOTE_INPUTS_DIR}/jfk_441khz.m4a tiny"

        cd $SOURCE_DIR/test
        adb -s $DEVICE push jfk_441khz.m4a $REMOTE_INPUTS_DIR/.
        adb -s $DEVICE shell $CMD
        exit 0 ;;
    *) 
        echo "Usage: "
        echo "  ${0} linux   : test for linux (in build_linux)"
        echo "  ${0} qnn|gpu : test for arm64 Android (QNN | GPU delegate in build_android)"
        echo "  ${0}         : test for arm64 Android (QNN delegate in build_android)"
        exit 1 ;;
esac
