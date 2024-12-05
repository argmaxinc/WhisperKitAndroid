#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting library & header files are copied into /libs & /inc folder
echo "Usage: "
echo "      ${0} x86  : build for x86 (in build_x86)"
echo "      ${0}      : build for arm64 Android (in build_android)"

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/../.build/ffmpeg"
BUILD_DIR=build

arg=$1
if [ -d "${SOURCE_DIR}/${BUILD_DIR}" ]; then
    rm -rf $SOURCE_DIR/$BUILD_DIR
fi

cd $SOURCE_DIR
CXXFLAGS="-std=c++17 ${CXXFLAGS}"

if [[ "$arg" == "x86" ]]; then
    echo "Now configuring FFmpeg for x86_64.."
    PLATFORM="x86"
    ARCH_CONFIG="--cc=gcc --cxx=g++ --enable-x86asm "
else
    echo "Now configuring FFmpeg for android arm64.."
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
  --prefix=${SOURCE_DIR}/${BUILD_DIR} \
  ${ARCH_CONFIG} \
  --libdir=${PWD}/build/lib \
  --incdir=${PWD}/build/include \
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

rm -rf ${BUILD_DIR}/lib/libavdevice.so*
rm -rf ${BUILD_DIR}/lib/libavfilter.so*
rm -rf ${BUILD_DIR}/include/libavdevice
rm -rf ${BUILD_DIR}/include/libavfilter

cp -rf ${BUILD_DIR}/lib/lib*.so* $SOURCE_DIR/../../libs/$PLATFORM/
cp -rf ${BUILD_DIR}/include/* $SOURCE_DIR/../../inc/
