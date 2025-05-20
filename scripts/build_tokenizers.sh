#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This build script runs when docker image is created.
# The resulting `libtokenizers_sys.so` is copied into /libs folder in the build.sh

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/../.source/tokenizers-sys"

PLATFORM=$1
if [ "$PLATFORM" = "" ]; then
    PLATFORM="android"
fi

# Install Rust using rustup (stable toolchain by default)
curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add cargo binaries to PATH
PATH="/root/.cargo/bin:${PATH}"

export ANDROID_NDK_HOME=/opt/android-ndk/android-ndk-r25c
export TOOLCHAIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64
export TARGET=aarch64-linux-android
export API=33

export CC="$TOOLCHAIN/bin/${TARGET}${API}-clang"
export AR="$TOOLCHAIN/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC"
export PATH=$PATH:$TOOLCHAIN/bin/

cd $SOURCE_DIR
if [ "$PLATFORM" = "android" ]; then
    rm $SOURCE_DIR/Cargo.lock
    
    source /root/.cargo/env

    TARGET=aarch64-linux-android

    rustup target add $TARGET

    cargo build --release --target $TARGET
    cp ${SOURCE_DIR}/target/$TARGET/release/*.so $CURRENT_DIR/../external/libs/$PLATFORM/
    cp $TOOLCHAIN/sysroot/usr/lib/$TARGET/libc++_shared.so $CURRENT_DIR/../external/libs/$PLATFORM/
else
    cargo build --release
    cp -rf ${SOURCE_DIR}/target/release/libtokenizers_sys.so $CURRENT_DIR/../external/libs/$PLATFORM/
fi
