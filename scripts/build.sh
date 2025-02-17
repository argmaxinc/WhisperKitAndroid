#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.


ARG=$1
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
SDL3_DIR="$SOURCE_DIR/.source/SDL"

case $ARG in
    "clean")
        echo "  ${0} clean: cleaning build files"
        if [ -d "$SOURCE_DIR/build" ]; then
            rm -rf $SOURCE_DIR/build
        fi

        if [ "$2" = "all" ]; then
            rm -rf $SOURCE_DIR/external
        fi

        exit 0 ;;

    "linux")
        echo "  ${0} linux   : building for linux (in build/linux)"
        PLATFORM="linux" ;;

    "gpu" | "qnn" | "" )
        echo "  ${0} [gpu|qnn] : building for arm64 Android (in build/android)"
        PLATFORM="android" ;;

    "jni" )
        echo "  ${0} jni : building for arm64 Android (in build/android)" 
        PLATFORM="android" ;;

    *)
        echo "Usage: "
        echo "  ${0} clean   : clean build files"
        echo "  ${0} linux   : build for x86 (in build/linux)"
        echo "  ${0} qnn|gpu : build for arm64 Android (QNN | GPU delegate in build/android)"
        echo "  ${0} jni     : build a shared object for arm64 Android (QNN) which can be used in an android app"
        echo "  ${0}         : build for arm64 Android (QNN delegate in build/android)"
        exit 1 ;;
esac

BUILD_DIR="build/${PLATFORM}"

# check if external directories exist
if [ ! -d $SOURCE_DIR/external/build/$PLATFORM ]; then
    mkdir -p $SOURCE_DIR/external/build/$PLATFORM
fi
if [ ! -d $SOURCE_DIR/external/libs/$PLATFORM ]; then
    mkdir -p $SOURCE_DIR/external/libs/$PLATFORM
fi
if [ ! -d $SOURCE_DIR/external/inc ]; then
    mkdir -p $SOURCE_DIR/external/inc
fi

# check if libtensorflowlite.so and its headers are built and installed
if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libtensorflowlite.so ]; then
    $SOURCE_DIR/scripts/build_tensorflow.sh $PLATFORM
elif [ ! -d $SOURCE_DIR/external/inc/flatbuffers ]; then
    $SOURCE_DIR/scripts/build_tensorflow.sh $PLATFORM
fi

# check if libSDL3.so is built and exists
if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libSDL3.so ]; then
    echo "SDL3 libs are not found, building it now.."
    $SOURCE_DIR/scripts/build_SDL.sh $PLATFORM
fi

# check if ffmpeg libs is built and exists
if [ ! -f $SOURCE_DIR/external/libs/$PLATFORM/libavcodec.so ]; then
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
    find "$TENSORFLOW_SOURCE_DIR/" $TENSORFLOW_SOURCE_DIR/bazel-bin/ \
        -name libtensorflowlite_gpu_delegate.so -exec cp {} $SOURCE_DIR/external/libs/android/ \;

    if [ "$ARG" = "gpu" ]; then # Generic TFLite GPU delegate
        rm $SOURCE_DIR/external/libs/android/libQnn*.so
        rm $SOURCE_DIR/external/libs/android/libqnn*.so
        rm $SOURCE_DIR/external/inc/QnnTFLiteDelegate.h
        QNN_DELEGATE="-DQNN_DELEGATE=0"
    else # QCOM QNN delegate (INCLUDES JNI BUILD)
        cp ${QNN_RUNTIME_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/external/libs/android/
        cp ${QNN_SDK_ROOT}/jni/arm64-v8a/lib*.so $SOURCE_DIR/external/libs/android/
        cp ${QNN_SDK_ROOT}/headers/QNN/QnnTFLiteDelegate.h $SOURCE_DIR/external/inc/
        QNN_DELEGATE="-DQNN_DELEGATE=1"
    fi

    if [ "$ARG" = "jni" ]; then # embed SDL3 in .so
        SDL3_FLAG="-DSDL3_DIR=${SDL3_DIR}"
        JNI_FLAG="-DJNI=1"
    else 
        SDL3_FLAG=""
        JNI_FLAG="-DJNI=0"
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
    ${SDL3_FLAG} \
    ${JNI_FLAG} \
    ${QNN_DELEGATE}
fi

echo "*****************"
echo "cmake is done.. "
echo "To build: cd ${SOURCE_DIR}/${BUILD_DIR}; ninja -j 12"
echo "Running build now..."
echo "*****************"

if [ ! -d "${SOURCE_DIR}/${BUILD_DIR}" ]; then
    mkdir ${SOURCE_DIR}/build/${PLATFORM}
fi
cd ${SOURCE_DIR}/${BUILD_DIR}
ninja -j 12
