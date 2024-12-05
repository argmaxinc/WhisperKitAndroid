#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# This script builds and runs `android-ndk-qnn-tensorflow-image` docker image with all dependencies
# As part of build process, the script downloads all dependencies into the .build/ folder.
# You will need `aria2` installed (see https://formulae.brew.sh/formula/aria2)

IMAGE_NAME="android-ndk-qnn-tensorflow-image"
CONTAINER_NAME="axie_tflite"
FORCE_REBUILD=false
FORCE_REMOVE=false

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

while getopts "rf" opt; do
  case ${opt} in
    r ) FORCE_REBUILD=true ;;
    f ) FORCE_REMOVE=true ;;
    \? ) echo "Usage: cmd [-r] [-f]"
         exit 1 ;;
  esac
done

set -e  # Exit on error

# Build the image only if it doesn't exist or if forced
if ! $(docker image inspect $IMAGE_NAME > /dev/null 2>&1) || $FORCE_REBUILD; then

  if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" != "" ]]; then
    docker ps -q --filter "ancestor=$IMAGE_NAME" | xargs -r docker stop
    docker ps -a -q --filter "ancestor=$IMAGE_NAME" | xargs -r docker rm
    docker rmi $IMAGE_NAME
  fi

  # Set Aria options to download using 8 connections
  ARIA_OPTIONS="-x 8 -s 8 --continue --file-allocation=none"

  BUILD_DIR="$SOURCE_DIR/.build"
  echo "Checking and retrieving dependencies..."
  if command -v aria2c &> /dev/null; then
    aria2c $ARIA_OPTIONS -d $BUILD_DIR https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-installer-linux-x86_64.sh
    aria2c $ARIA_OPTIONS -d $BUILD_DIR https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
    aria2c $ARIA_OPTIONS -d $BUILD_DIR https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
    aria2c $ARIA_OPTIONS -d $BUILD_DIR https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.27.0/qnn-runtime-2.27.0.aar
    aria2c $ARIA_OPTIONS -d $BUILD_DIR https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-tflite-delegate/2.27.0/qnn-tflite-delegate-2.27.0.aar
  else
    echo "Missing aria2c. Install using 'brew install aria2'. See https://formulae.brew.sh/formula/aria2"
    exit 0
  fi
  if [ ! -d "$BUILD_DIR/tensorflow" ]; then
    echo "Cloning tensorflow..."
    git clone --depth 1 --branch v2.16.2 https://github.com/tensorflow/tensorflow.git "$BUILD_DIR/tensorflow"
  fi
  if [ ! -d "$BUILD_DIR/ffmpeg" ]; then
    echo "Cloning ffmpeg..."
    git clone --branch release/7.1 https://github.com/FFmpeg/FFmpeg.git "$BUILD_DIR/ffmpeg"
  fi
  if [ ! -d "$BUILD_DIR/SDL" ]; then
    echo "Cloning SDL3..."
    git clone https://github.com/libsdl-org/SDL.git "$BUILD_DIR/SDL"
  fi

  echo "Building Docker image: $IMAGE_NAME"
  docker build --platform=linux/amd64 -t $IMAGE_NAME -f "$CURRENT_DIR/Dockerfile" $SOURCE_DIR
else
  echo "Docker image $IMAGE_NAME already exists."
fi

# Check if the container exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  if $FORCE_REMOVE; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
  else
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Starting existing container: $CONTAINER_NAME"
    docker start $CONTAINER_NAME
    fi
    echo "SSHing into existing container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME /bin/bash -c "echo 'Environment Variables:' && printenv && exec /bin/bash"
    exit 0
  fi
fi

# Run a new container
echo "Starting new container: $CONTAINER_NAME"
docker run -it --name $CONTAINER_NAME \
  --mount type=bind,source=$SOURCE_DIR,target=/src/AXIE \
  $IMAGE_NAME /bin/bash -c "echo 'Environment Variables:' && printenv && exec /bin/bash"
