#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

echo "Usage: "
echo "      ${0} clean: clean build files"
echo "      ${0}      : mkdir build and cmake to it"

cmd=$1
if [[ "$cmd" == "clean" ]]; then
    if [ -d build ]; then
        cd build
        ninja clean
        rm CMakeCache.txt
        rm -rf CMakeFiles
        rm -rf cmake_install.cmake
        rm -rf Makefile
        rm -rf compile_commands.json
        find . -name \CMakeCache.txt -type f -delete
        rm build.ninja
        rm axie_tflite
    fi
    exit 0
fi

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

mkdir -p $SOURCE_DIR/libs
find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ -name libtensorflowlite_gpu_delegate.so -exec cp {} $SOURCE_DIR/libs/ \;
cp ${QNN_RUNTIME_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/
cp ${QNN_SDK_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/
cp ${QNN_SDK_ROOT}/headers/QNN/QnnTFLiteDelegate.h $SOURCE_DIR/inc/

cmake \
-H$SOURCE_DIR \
-DCMAKE_SYSTEM_NAME=Android \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_SYSTEM_VERSION=33 \
-DANDROID_PLATFORM=android-33 \
-DANDROID_ABI=arm64-v8a \
-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
-DANDROID_NDK=${ANDROID_NDK_ROOT} \
-DCMAKE_ANDROID_NDK=${ANDROID_NDK_ROOT} \
-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
-DCMAKE_MAKE_PROGRAM=${ANDROID_HOME}/cmake/3.22.1/bin/ninja \
-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SOURCE_DIR/build \
-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SOURCE_DIR/build \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-B$SOURCE_DIR/build \
-GNinja \
-DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR} \
-DQNN_SDK_ROOT=${QNN_SDK_ROOT}

echo "*****************"
echo "cmake is done.. "
echo "To build: cd ${SOURCE_DIR}/build; ninja -j 12"
echo "Running build now..."
echo "*****************"

cd ${SOURCE_DIR}/build
ninja -j 12
