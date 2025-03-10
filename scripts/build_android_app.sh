#!/bin/bash

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
APP_SOURCE_DIR="$SOURCE_DIR/androidApp/app/src/main"

if [ ! -d "$APP_SOURCE_DIR/assets/" ]; then
    mkdir -p "$APP_SOURCE_DIR/assets/"
fi
cp "$SOURCE_DIR/test/"*.m4a "$APP_SOURCE_DIR/assets/"

if [ ! -d "$APP_SOURCE_DIR/jniLibs/arm64-v8a/" ]; then
    mkdir -p "$APP_SOURCE_DIR/jniLibs/arm64-v8a/"
fi
cp "$SOURCE_DIR/external/libs/android/"*.so "$APP_SOURCE_DIR/jniLibs/arm64-v8a/"
cp "$SOURCE_DIR/build/android/"*.so "$APP_SOURCE_DIR/jniLibs/arm64-v8a/"


