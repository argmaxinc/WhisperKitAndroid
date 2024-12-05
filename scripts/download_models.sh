#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

command -v aria2c &> /dev/null || { echo >&2 ""Missing aria2c. Install using 'brew install aria2'. See https://formulae.brew.sh/formula/aria2""; exit 1; }

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

# Set Aria options to download using 8 connections
ARIA_OPTIONS="-x 8 -s 8 --continue --file-allocation=none"

# Set directories
TINY_MODELS_DIR="$SOURCE_DIR/openai_whisper-tiny"
BASE_MODELS_DIR="$SOURCE_DIR/openai_whisper-base"
SMALL_MODELS_DIR="$SOURCE_DIR/openai_whisper-small"

# Make sure folders exist
mkdir -p "$TINY_MODELS_DIR"
mkdir -p "$BASE_MODELS_DIR"
mkdir -p "$SMALL_MODELS_DIR"

# Download Whisper auxiliary models
HF_ARGMAX_URL="https://huggingface.co/argmaxinc/whisperkit-android/resolve/main"

aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o converted_vocab.json $HF_ARGMAX_URL/converted_vocab.json
aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o MelSpectrogram.tflite $HF_ARGMAX_URL/melspectrogram.tflite
aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o postproc.tflite $HF_ARGMAX_URL/postproc.tflite
aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o voice_activity_detection.tflite $HF_ARGMAX_URL/voice_activity_detection.tflite
cp $TINY_MODELS_DIR/* $BASE_MODELS_DIR/.
cp $TINY_MODELS_DIR/* $SMALL_MODELS_DIR/.

# Download Qualcomm models
HF_QUALCOMM_URL="https://huggingface.co/qualcomm"

aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Tiny-En/resolve/main/WhisperDecoder.tflite
aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Tiny-En/resolve/main/WhisperEncoder.tflite
aria2c $ARIA_OPTIONS -d "$BASE_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Base-En/resolve/main/WhisperDecoder.tflite
aria2c $ARIA_OPTIONS -d "$BASE_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Base-En/resolve/main/WhisperEncoder.tflite
aria2c $ARIA_OPTIONS -d "$SMALL_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Small-En/resolve/main/WhisperDecoder.tflite
aria2c $ARIA_OPTIONS -d "$SMALL_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Small-En/resolve/main/WhisperEncoder.tflite
