#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

command -v aria2c &> /dev/null || { echo >&2 ""Missing aria2c. Install using 'brew install aria2'. See https://formulae.brew.sh/formula/aria2""; exit 1; }

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
MODELS_DIR="$SOURCE_DIR/models"
# Set Aria options to download using 8 connections
ARIA_OPTIONS="-x 8 -s 8 --continue --file-allocation=none"

# Set directories
TINY_MODELS_DIR="$MODELS_DIR/openai_whisper-tiny"
BASE_MODELS_DIR="$MODELS_DIR/openai_whisper-base"
SMALL_MODELS_DIR="$MODELS_DIR/openai_whisper-small"

function SAFE_MODEL_DIRECTORY(){
    if [ ! -d "${1}" ]; then
        echo "mkdir ${1} .."
        mkdir -p "${1}"
    fi
}

SAFE_MODEL_DIRECTORY $TINY_MODELS_DIR
SAFE_MODEL_DIRECTORY $BASE_MODELS_DIR
SAFE_MODEL_DIRECTORY $SMALL_MODELS_DIR

# Download Whisper auxiliary models
HF_ARGMAX_URL="https://huggingface.co/argmaxinc/whisperkit-litert/resolve/main"
HF_OPENAI_TINY_URL="https://huggingface.co/openai/whisper-tiny.en/resolve/main"

if [ ! -f $TINY_MODELS_DIR/tokenizer.json ]; then
    aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o tokenizer.json $HF_OPENAI_TINY_URL/tokenizer.json
    aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o MelSpectrogram.tflite $HF_ARGMAX_URL/quic_openai_whisper-tiny.en/MelSpectrogram.tflite
fi
if [ ! -f $BASE_MODELS_DIR/tokenizer.json ]; then
    cp $TINY_MODELS_DIR/tokenizer.json $BASE_MODELS_DIR/.
    cp $TINY_MODELS_DIR/MelSpectrogram.tflite $BASE_MODELS_DIR/.
fi
if [ ! -f $SMALL_MODELS_DIR/tokenizer.json ]; then
    cp $TINY_MODELS_DIR/tokenizer.json $SMALL_MODELS_DIR/.
    cp $TINY_MODELS_DIR/MelSpectrogram.tflite $SMALL_MODELS_DIR/.
fi

# Download Qualcomm models
HF_QUALCOMM_URL="https://huggingface.co/qualcomm"

if [ ! -f $TINY_MODELS_DIR/TextDecoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Tiny-En/resolve/main/WhisperDecoder.tflite
fi
if [ ! -f $TINY_MODELS_DIR/AudioEncoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$TINY_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Tiny-En/resolve/main/WhisperEncoder.tflite
fi
if [ ! -f $BASE_MODELS_DIR/TextDecoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$BASE_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Base-En/resolve/main/WhisperDecoder.tflite
fi
if [ ! -f $BASE_MODELS_DIR/AudioEncoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$BASE_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Base-En/resolve/main/WhisperEncoder.tflite
fi
if [ ! -f $SMALL_MODELS_DIR/TextDecoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$SMALL_MODELS_DIR" -o TextDecoder.tflite $HF_QUALCOMM_URL/Whisper-Small-En/resolve/main/WhisperDecoder.tflite
fi
if [ ! -f $SMALL_MODELS_DIR/AudioEncoder.tflite ]; then
    aria2c $ARIA_OPTIONS -d "$SMALL_MODELS_DIR" -o AudioEncoder.tflite $HF_QUALCOMM_URL/Whisper-Small-En/resolve/main/WhisperEncoder.tflite
fi
