#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting `libtensorflowlite_gpu_delegate.so` is copied into /libs folder in the build.sh

export PYTHON_BIN_PATH=/usr/bin/python3
export PYTHON_LIB_PATH=/usr/lib/python3/dist-packages
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_CLANG=1
export CLANG_COMPILER_PATH=/usr/bin/clang
export CC_OPT_FLAGS=-Wno-sign-compare
export TF_SET_ANDROID_WORKSPACE=1
export ANDROID_NDK_API_LEVEL=24
export ANDROID_API_LEVEL=34
export ANDROID_BUILD_TOOLS_VERSION=34.0.0

cd $TENSORFLOW_SOURCE_DIR && ./configure

echo "Building libtensorflowlite_gpu_delegate.so ..."
printenv
mkdir -p tensorflow/lite/delegates/gpu
bazel build -c opt --config android_arm64 --cxxopt=--std=c++17 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
