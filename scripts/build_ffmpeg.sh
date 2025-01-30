#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting library & header files are copied into external/libs & external/inc folder
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/../.source/ffmpeg"
PLATFORM=$1
if [ "$PLATFORM" = "" ]; then
    PLATFORM="android"
fi
BUILD_DIR=$CURRENT_DIR/../external/build/$PLATFORM/ffmpeg

cd $SOURCE_DIR
CXXFLAGS="-std=c++17 ${CXXFLAGS}"

if [ "$PLATFORM" = "linux" ]; then
    echo "  ${0} linux   : build for linux (in build_linux)"
    PLATFORM="linux"
    ARCH_CONFIG="--cc=gcc --cxx=g++ --enable-x86asm "
else
    echo "  ${0} android : build for arm64 Android (in build/android)"
    PLATFORM="android"
    ARCH_CONFIG="--cross-prefix=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33- \
    --sysroot=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    --enable-cross-compile \
    --target-os=android \
    --arch=arm64 \
    --cpu=armv8-a \
    --nm=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm \
    --ar=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar \
    --strip=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip \
    --cc=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang \
    --cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang++ \
    --disable-x86asm "
fi

./configure \
  --prefix=${BUILD_DIR} \
  ${ARCH_CONFIG} \
  --extra-cflags="$CFLAGS" \
  --extra-cxxflags="$CFLAGS" \
  --extra-ldflags=-ldl \
  --disable-programs \
  --disable-logging \
  --disable-everything \
  --disable-ffplay \
  --disable-doc \
  --disable-devices \
  --disable-swscale \
  --disable-hwaccels \
  --disable-parsers \
  --disable-bsfs \
  --disable-debug \
  --disable-indevs \
  --disable-outdevs \
  --disable-static \
  --enable-ffmpeg \
  --enable-ffprobe \
  --enable-avformat \
  --enable-avcodec \
  --enable-swresample \
  --enable-decoder="mov,mp4,aac,mp3,m4a,flac,vorbis,wavpack" \
  --enable-parser="mov,mp4,aac,mp3,m4a,flac,ogg,wav" \
  --enable-demuxer="mov,mp4,aac,mp3,m4a,flac,ogg,wav" \
  --enable-optimizations \
  --enable-stripping \
  --enable-small \
  --enable-shared \
  --enable-protocol=file,http,tcp,rtmp,rtsp

sleep 1
make clean; make -j 12
sleep 1
make install

cp -rf ${BUILD_DIR}/lib/lib*.so* $CURRENT_DIR/../external/libs/$PLATFORM/
cp -rf ${BUILD_DIR}/include/* $CURRENT_DIR/../external/inc/
