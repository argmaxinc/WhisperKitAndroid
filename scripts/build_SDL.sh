#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting `libSDL3.so` is copied into /libs folder in the build.sh
echo "Usage: "
echo "      ${0} x86  : build for x86 (in build_x86)"
echo "      ${0}      : build for arm64 Android (in build_android)"

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/../.build/SDL"
BUILD_DIR=build

arg=$1
if [ -d "${SOURCE_DIR}/${BUILD_DIR}" ]; then
    rm -rf $SOURCE_DIR/$BUILD_DIR
fi

if [[ "$arg" == "x86" ]]; then
    PLATFORM="x86"
    cmake \
    -H$SOURCE_DIR \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja 
else
    PLATFORM="android"
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
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja 
fi

echo "*****************"
echo "cmake is done.. "
echo "To build: cd ${SOURCE_DIR}/${BUILD_DIR}; ninja -j 12"
echo "Running build now..."
echo "*****************"


cd ${SOURCE_DIR}/${BUILD_DIR}
ninja -j 12

cp -rf libSDL3.so* $SOURCE_DIR/../../libs/$PLATFORM/
cp -rf ../include/SDL3 $SOURCE_DIR/../../inc/SDL3
