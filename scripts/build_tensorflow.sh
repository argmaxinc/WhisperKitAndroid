#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting `libtensorflowlite_gpu_delegate.so` is copied into /libs folder in the build.sh

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
PLATFORM=$1
if [ "$PLATFORM" = "" ]; then
    PLATFORM="android"
fi

export PYTHON_BIN_PATH=/usr/bin/python3
export PYTHON_LIB_PATH=/usr/lib/python3/dist-packages
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_CLANG=1
export CLANG_COMPILER_PATH=/usr/bin/clang-18
export CC_OPT_FLAGS=-Wno-sign-compare

# nightly tf commit needs bazel 7.4.1
USING_NIGHTLY_TF_COMMIT=1
if [ "$USING_NIGHTLY_TF_COMMIT" = "1" ]; then
    cd "/usr/local/lib/bazel/bin" && curl -fLO https://releases.bazel.build/7.4.1/release/bazel-7.4.1-linux-x86_64 && chmod +x bazel-7.4.1-linux-x86_64
fi

if [ "$PLATFORM" = "android" ]; then
    export TF_SET_ANDROID_WORKSPACE=1
    export ANDROID_NDK_API_LEVEL=24
    export ANDROID_API_LEVEL=34
    export ANDROID_BUILD_TOOLS_VERSION=34.0.0

    cd $TENSORFLOW_SOURCE_DIR && ./configure

    if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libtensorflowlite_gpu_delegate.so ]; then
        echo "$SOURCE_DIR/external/libs/$PLATFORM ..."
        echo "Building libtensorflowlite_gpu_delegate.so ..."
        printenv
        mkdir -p tensorflow/lite/delegates/gpu
        bazel build -c opt --config android_arm64 --cxxopt=--std=c++17 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
        find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ \
            -name libtensorflowlite_gpu_delegate.so -exec cp {} $SOURCE_DIR/external/libs/android/ \;
    fi

    if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libtensorflowlite.so ]; then
        bazel build -c opt --config android_arm64 --cxxopt=--std=c++17 //tensorflow/lite:libtensorflowlite.so
        find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ \
            -name libtensorflowlite.so -exec cp {} $SOURCE_DIR/external/libs/$PLATFORM/ \;
    fi
else
    export TF_SET_ANDROID_WORKSPACE=0
    if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libtensorflowlite.so ]; then
        cd $TENSORFLOW_SOURCE_DIR && ./configure

        bazel build //tensorflow/lite:libtensorflowlite.so
        find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ \
            -name libtensorflowlite.so -exec cp {} $SOURCE_DIR/external/libs/$PLATFORM/ \;
    fi
fi

if [ ! -d $SOURCE_DIR/external/inc/flatbuffers ]; then
    cp -rf $TENSORFLOW_SOURCE_DIR/bazel-tensorflow/external/flatbuffers/include/flatbuffers \
        $SOURCE_DIR/external/inc/.
fi
