#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting `libSDL3.so` and header files are copied into /libs & /inc folder
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/../.build/SDL"
PLATFORM=$1
if [ "$PLATFORM" = "" ]; then
    PLATFORM="android"
fi
BUILD_DIR=$CURRENT_DIR/../external_build/$PLATFORM/SDL

if [ "$PLATFORM" = "linux" ]; then
    echo "  ${0} linux   : build for linux (in ${BUILD_DIR})"
    cmake \
    -H$SOURCE_DIR \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$BUILD_DIR \
    -GNinja 
else
    echo "  ${0} android : build for arm64 Android (in ${BUILD_DIR})"
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
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$BUILD_DIR \
    -GNinja 
fi

echo "*****************"
echo "cmake is done.. "
echo "To build: cd ${BUILD_DIR}; ninja -j 12"
echo "Running build now..."
echo "*****************"


cd ${BUILD_DIR}
ninja -j 12

cp -rf libSDL3.so* $CURRENT_DIR/../libs/$PLATFORM/
cd $SOURCE_DIR
cp -rf include/SDL3 $CURRENT_DIR/../inc/SDL3
