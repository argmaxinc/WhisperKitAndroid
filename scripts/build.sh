#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

echo "Usage: "
echo "      ${0} clean: clean build files"
echo "      ${0} x86  : build for x86 (in build_x86)"
echo "      ${0} gpu  : build for arm64 Android (in build_android)"
echo "      ${0}      : build for Android with QNN (in build_android)"

arg=$1
if [[ "$arg" == "clean" ]]; then
    if [ -d "build_android" ]; then
        cd build_android
    else
     	cd build_x86
    fi
    ninja clean
    rm CMakeCache.txt
    rm -rf CMakeFiles
    rm -rf cmake_install.cmake
    rm -rf Makefile
    rm -rf compile_commands.json
    find . -name \CMakeCache.txt -type f -delete
    rm build.ninja
    exit 0
fi

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

if [[ "$arg" == "x86" ]]; then
    PLATFORM="x86"
else
    PLATFORM="android"
fi
BUILD_DIR="build_${PLATFORM}"

# check if libSDL3.so is built and exists
if [ ! -f $SOURCE_DIR/libs/$PLATFORM/libSDL3.so ]; then
    echo "SDL3 libs are not found, building it now.."
    ./build_SDL.sh $arg
fi

if [ ! -f $SOURCE_DIR/libs/$PLATFORM/libavcodec.so ]; then
    echo "ffmpeg libs are not found, building it now.."
    ./build_ffmpeg.sh $arg
fi


if [[ "$arg" == "x86" ]]; then
    cmake \
    -H$SOURCE_DIR \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja \
    -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR}
else
    CURRENT_DIR="$(dirname "$(realpath "$0")")"
    SOURCE_DIR="$CURRENT_DIR/.."

    find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ -name libtensorflowlite_gpu_delegate.so -exec cp {} $SOURCE_DIR/libs/android/ \;

    if [[ "$arg" == "gpu" ]]; then # Generic TFLite GPU delegate
        QNN_OR_GPU="-DGPU_DELEGATE=True"
    else # QCOM QNN delegate
        cp ${QNN_RUNTIME_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/android/
        cp ${QNN_SDK_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/android/
        cp ${QNN_SDK_ROOT}/headers/QNN/QnnTFLiteDelegate.h $SOURCE_DIR/inc/
        QNN_OR_GPU="-DQNN_DELEGATE=True"       
    fi

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
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja \
    -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR} \
    ${QNN_OR_GPU}
fi

echo "*****************"
echo "cmake is done.. "
echo "To build: cd ${SOURCE_DIR}/${BUILD_DIR}; ninja -j 12"
echo "Running build now..."
echo "*****************"

if [ ! -d "${SOURCE_DIR}/${BUILD_DIR}" ]; then
    mkdir ${SOURCE_DIR}/${BUILD_DIR}
fi
cd ${SOURCE_DIR}/${BUILD_DIR}
ninja -j 12
