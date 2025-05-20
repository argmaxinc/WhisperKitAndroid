# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.
FROM ubuntu:22.04 AS build-env

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless build-essential cmake git \
    python3 python3-pip apt-transport-https \
    curl wget vim g++ unzip libdrm-dev \
    pkg-config gnome-desktop-testing libasound2-dev libpulse-dev \
    libxrandr-dev libxcursor-dev libxfixes-dev libxi-dev libxss-dev \
    libxkbcommon-dev libgbm-dev libgl-dev libgles2-mesa-dev \
    libegl1-mesa-dev libdbus-1-dev libibus-1.0-dev libudev-dev fcitx-libs-dev \
    libpipewire-0.3-dev libwayland-dev libdecor-0-dev nasm yasm libssl-dev \
    lsb-release software-properties-common gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

FROM build-env AS tflite-env

ENV TF_PYTHON_VERSION=3.10

ARG CMAKE_VERION=3.22.1
ARG CMD_TOOLS=cmdline-tools
ENV ANDROID_SDK_HOME=/opt/android-sdk

# ARG ANDROID_NDK_VERSION=android-ndk-r27b  => this is not supported by tensorflow
ARG ANDROID_NDK_VERSION=android-ndk-r25c
ARG ANDROID_NDK_PATH=/opt/android-ndk
ENV ANDROID_NDK_HOME=$ANDROID_NDK_PATH/$ANDROID_NDK_VERSION
ENV TENSORFLOW_SOURCE_DIR=/opt/tensorflow

ENV ANDROID_NDK_ROOT=$ANDROID_NDK_PATH/$ANDROID_NDK_VERSION
ENV ANDROID_HOME=$ANDROID_SDK_HOME

ENV QNN_SDK_ROOT=/opt/qnn-sdk
ENV QNN_RUNTIME_ROOT=/opt/qnn-runtime
ENV AXIE_ROOT=/src/AXIE

ARG ANDROID_NDK_ZIP=$ANDROID_NDK_VERSION-linux.zip
ARG BAZEL_INSTALLER=bazel-6.5.0-installer-linux-x86_64.sh
ARG BAZEL_DIR=/opt/bazel
ARG QNN_RUNTIME=qnn-runtime-2.33.0.aar
ARG QNN_TFLITE_DELEGATE=qnn-litert-delegate-2.33.0.aar
ARG ANDROID_COMMAND_LINE_TOOLS=commandlinetools-linux-11076708_latest.zip

# Copy build dependencies
ADD .source/$ANDROID_NDK_ZIP $ANDROID_NDK_PATH/
ADD .source/$BAZEL_INSTALLER $BAZEL_DIR/$BAZEL_INSTALLER
ADD .source/$QNN_RUNTIME $QNN_RUNTIME_ROOT/
ADD .source/$QNN_TFLITE_DELEGATE $QNN_SDK_ROOT/
ADD .source/$ANDROID_COMMAND_LINE_TOOLS $ANDROID_SDK_HOME/

# install clang-18
RUN wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- 18 && \
    ln -s /usr/bin/clang-18 /usr/bin/clang

# Unzip and install dependencies
RUN cd $QNN_RUNTIME_ROOT && unzip $QNN_RUNTIME && rm $QNN_RUNTIME && \
    cd $QNN_SDK_ROOT && unzip $QNN_TFLITE_DELEGATE && rm $QNN_TFLITE_DELEGATE && \
    cd $ANDROID_NDK_PATH && unzip $ANDROID_NDK_ZIP && rm $ANDROID_NDK_ZIP && \
    cd $ANDROID_SDK_HOME && unzip $ANDROID_COMMAND_LINE_TOOLS && rm $ANDROID_COMMAND_LINE_TOOLS && \
    yes | $CMD_TOOLS/bin/sdkmanager --install "platform-tools" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "platforms;android-34" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "build-tools;34.0.0" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "cmake;$CMAKE_VERION" --sdk_root=$ANDROID_SDK_HOME && \
    ln -s $AXIE_ROOT/.source/tensorflow $TENSORFLOW_SOURCE_DIR && \
    chmod +x $BAZEL_DIR/$BAZEL_INSTALLER && $BAZEL_DIR/$BAZEL_INSTALLER && rm -rf $BAZEL_DIR

# Set up the PATH
ENV PATH=$PATH:$ANDROID_NDK_HOME:$ANDROID_SDK_HOME/$CMD_TOOLS/bin:$ANDROID_SDK_HOME/platform-tools:$ANDROID_SDK_HOME/cmake/$CMAKE_VERION/bin

WORKDIR $AXIE_ROOT
