# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

FROM ubuntu:22.04 AS build-env

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless build-essential cmake git \
    python3 python3-pip apt-transport-https \
    curl wget vim g++ unzip clang \
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

ARG ANDROID_NDK_ZIP=$ANDROID_NDK_VERSION-linux.zip
ARG BAZEL_INSTALLER=bazel-6.5.0-installer-linux-x86_64.sh
ARG BAZEL_DIR=/opt/bazel
ARG QNN_RUNTIME=qnn-runtime-2.27.0.aar
ARG QNN_TFLITE_DELEGATE=qnn-tflite-delegate-2.27.0.aar
ARG ANDROID_COMMAND_LINE_TOOLS=commandlinetools-linux-11076708_latest.zip

# Copy build dependencies
ADD .build/tensorflow $TENSORFLOW_SOURCE_DIR/
ADD scripts/build_tensorflow.sh /tmp/
ADD .build/$ANDROID_NDK_ZIP $ANDROID_NDK_PATH/
ADD .build/$BAZEL_INSTALLER $BAZEL_DIR/$BAZEL_INSTALLER
ADD .build/$QNN_RUNTIME $QNN_RUNTIME_ROOT/
ADD .build/$QNN_TFLITE_DELEGATE $QNN_SDK_ROOT/
ADD .build/$ANDROID_COMMAND_LINE_TOOLS $ANDROID_SDK_HOME/

# Unzip and install dependencies
RUN cd $QNN_RUNTIME_ROOT && unzip $QNN_RUNTIME && rm $QNN_RUNTIME && \
    cd $QNN_SDK_ROOT && unzip $QNN_TFLITE_DELEGATE && rm $QNN_TFLITE_DELEGATE && \
    cd $ANDROID_NDK_PATH && unzip $ANDROID_NDK_ZIP && rm $ANDROID_NDK_ZIP && \
    cd $ANDROID_SDK_HOME && unzip $ANDROID_COMMAND_LINE_TOOLS && rm $ANDROID_COMMAND_LINE_TOOLS && \
    yes | $CMD_TOOLS/bin/sdkmanager --install "platform-tools" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "platforms;android-34" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "build-tools;34.0.0" --sdk_root=$ANDROID_SDK_HOME && \
    $CMD_TOOLS/bin/sdkmanager --install "cmake;$CMAKE_VERION" --sdk_root=$ANDROID_SDK_HOME && \
    chmod +x $BAZEL_DIR/$BAZEL_INSTALLER && $BAZEL_DIR/$BAZEL_INSTALLER && rm $BAZEL_DIR/$BAZEL_INSTALLER
RUN chmod +x /tmp/build_tensorflow.sh && /tmp/build_tensorflow.sh

# Set up the PATH
ENV PATH=$PATH:$ANDROID_NDK_HOME:$TENSORFLOW_SOURCE_DIR:$ANDROID_SDK_HOME/$CMD_TOOLS/bin:$ANDROID_SDK_HOME/platform-tools:$ANDROID_SDK_HOME/cmake/$CMAKE_VERION/bin

WORKDIR /src/AXIE
