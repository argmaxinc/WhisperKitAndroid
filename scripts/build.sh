#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.


ARG=$1
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

case $ARG in
    "clean")
        echo "  ${0} clean: cleaning build files"
        if [ -d "$SOURCE_DIR/build_android" ]; then
            rm -rf $SOURCE_DIR/build_android
        fi

        if [ -d "$SOURCE_DIR/build_linux" ]; then
            rm -rf $SOURCE_DIR/build_linux
        fi

        if [ "$2" = "all" ]; then
            rm -rf $SOURCE_DIR/libs
            rm -rf external_build
        fi

        exit 0 ;;

    "linux")
        echo "  ${0} linux   : building for linux (in build_linux)"
        PLATFORM="linux" ;;

    "gpu" | "qnn" | "" )
        echo "  ${0} [gpu|qnn] : building for arm64 Android (in build_android)"
        PLATFORM="android" ;;

    *) 
        echo "Usage: "
        echo "  ${0} clean   : clean build files"
        echo "  ${0} linux   : build for x86 (in build_linux)"
        echo "  ${0} qnn|gpu : build for arm64 Android (QNN | GPU delegate in build_android)"
        echo "  ${0}         : build for arm64 Android (QNN delegate in build_android)"
        exit 1 ;;
esac

mkdir -p $SOURCE_DIR/libs
mkdir -p $SOURCE_DIR/libs/$PLATFORM
BUILD_DIR="build_${PLATFORM}"

# check if libtensorflowlite.so and its headers are built and installed
if [ ! -f $SOURCE_DIR/libs/$PLATFORM/libtensorflowlite.so ]; then
    $SOURCE_DIR/scripts/build_tensorflow.sh $PLATFORM
elif [ ! -d $SOURCE_DIR/inc/flatbuffers ]; then
    $SOURCE_DIR/scripts/build_tensorflow.sh $PLATFORM
fi

# check if libSDL3.so is built and exists
if [ ! -f $SOURCE_DIR/libs/$PLATFORM/libSDL3.so ]; then
    echo "SDL3 libs are not found, building it now.."
    $SOURCE_DIR/scripts/build_SDL.sh $PLATFORM
fi

# check if ffmpeg libs is built and exists
if [ ! -f $SOURCE_DIR/libs/$PLATFORM/libavcodec.so ]; then
    echo "ffmpeg libs are not found, building it now.."
    $SOURCE_DIR/scripts/build_ffmpeg.sh $PLATFORM
fi

if [ -d "$SOURCE_DIR/$BUILD_DIR" ]; then
    cd $SOURCE_DIR/$BUILD_DIR
    ninja clean
fi

if [ "$ARG" = "linux" ]; then
    cmake \
    -H$SOURCE_DIR \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SOURCE_DIR/$BUILD_DIR \
    -DCMAKE_BUILD_TYPE=release \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja \
    -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR} 
else
    find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ -name libtensorflowlite_gpu_delegate.so -exec cp {} $SOURCE_DIR/libs/android/ \;

    if [ "$ARG" = "gpu" ]; then # Generic TFLite GPU delegate
        rm $SOURCE_DIR/libs/android/libQnn*.so
        rm $SOURCE_DIR/libs/android/libqnn*.so
        rm $SOURCE_DIR/inc/QnnTFLiteDelegate.h
        QNN_DELEGATE="-DQNN_DELEGATE=OFF"
    else # QCOM QNN delegate
        cp ${QNN_RUNTIME_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/android/
        cp ${QNN_SDK_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/libs/android/
        cp ${QNN_SDK_ROOT}/headers/QNN/QnnTFLiteDelegate.h $SOURCE_DIR/inc/
        QNN_DELEGATE="-DQNN_DELEGATE=ON"
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
    -DCMAKE_BUILD_TYPE=release \
    -B$SOURCE_DIR/$BUILD_DIR \
    -GNinja \
    -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR} \
    ${QNN_DELEGATE}
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
