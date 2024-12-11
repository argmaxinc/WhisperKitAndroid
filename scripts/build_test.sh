#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
ARG=$1

case $ARG in
    "linux")
        echo "  ${0} linux   : run in Docker"
        cd $SOURCE_DIR
        ./build_linux/whisperax_cli test/data/ted_60.m4a tiny
        exit 0 ;;

    "gpu" | "qnn" | "" )
        echo "  ${0} [gpu|qnn] : run on Host PC"
        cd $SOURCE_DIR/test
        pip install -r $SOURCE_DIR/test/requirements.txt
        python3 whisperkit_android_test.py -i $SOURCE_DIR/test/data -m tiny
        exit 0 ;;
    *) 
        echo "Usage: "
        echo "  ${0} linux   : test for linux (in build_linux)"
        echo "  ${0} qnn|gpu : test for arm64 Android (QNN | GPU delegate in build_android)"
        echo "  ${0}         : test for arm64 Android (QNN delegate in build_android)"
        exit 1 ;;
esac
