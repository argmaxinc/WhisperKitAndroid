#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.


ARG=$1
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

if [ "$ARG" = "jni" ]; then
    sleep 1
    cd ${SOURCE_DIR}
    files=(
        "external/libs/android/libavcodec.so"
        "external/libs/android/libavformat.so" 
        "external/libs/android/libavutil.so"
        "external/libs/android/libswresample.so"
        "external/libs/android/libtensorflowlite_gpu_delegate.so"
        "external/libs/android/libtensorflowlite.so"
        "external/libs/android/libtokenizers_sys.so"
        "build/android/libwhisperkit_jni.so"
        "build/android/libwhisperkit.so"
        "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    )
    
    # Destination directory
    DEST="$SOURCE_DIR/android/whisperkit/src/main/jniLibs/arm64-v8a"
    if [ ! -d "$DEST" ]; then
        mkdir -p $DEST
    fi
    for file in "${files[@]}"; do
        cp "$file" "$DEST"
    done
    chmod 755 $DEST/*.so
fi
