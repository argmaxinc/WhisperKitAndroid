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
QCOM_TINY_EN_MODELS_DIR="$MODELS_DIR/quic_openai_whisper-tiny.en"
QCOM_BASE_EN_MODELS_DIR="$MODELS_DIR/quic_openai_whisper-base.en"
QCOM_SMALL_EN_MODELS_DIR="$MODELS_DIR/quic_openai_whisper-small.en"

TINY_MODELS_DIR="$MODELS_DIR/openai_whisper-tiny"
BASE_MODELS_DIR="$MODELS_DIR/openai_whisper-base"
TINY_EN_MODELS_DIR="$MODELS_DIR/openai_whisper-tiny.en"
BASE_EN_MODELS_DIR="$MODELS_DIR/openai_whisper-base.en"

# Download Whisper auxiliary models
HF_ARGMAX_URL="https://huggingface.co/argmaxinc/whisperkit-litert/resolve/main"
# Download Qualcomm models
HF_QUALCOMM_URL="https://huggingface.co/qualcomm"

HF_OPENAI_TINY_EN_URL="https://huggingface.co/openai/whisper-tiny.en/resolve/main"
HF_OPENAI_TINY_URL="https://huggingface.co/openai/whisper-tiny/resolve/main"
HF_OPENAI_BASE_EN_URL="https://huggingface.co/openai/whisper-base.en/resolve/main"
HF_OPENAI_BASE_URL="https://huggingface.co/openai/whisper-base/resolve/main"
HF_OPENAI_SMALL_EN_URL="https://huggingface.co/openai/whisper-small.en/resolve/main"
HF_OPENAI_SMALL_EN_URL="https://huggingface.co/openai/whisper-small/resolve/main"

HF_QCOM_TINY_EN_URL=$HF_QUALCOMM_URL/Whisper-Tiny-En/resolve/main
HF_QCOM_BASE_EN_URL=$HF_QUALCOMM_URL/Whisper-Base-En/resolve/main
HF_QCOM_SMALL_EN_URL=$HF_QUALCOMM_URL/Whisper-Small-En/resolve/main

HF_AX_TINY_EN_URL=$HF_ARGMAX_URL/openai_whisper-tiny.en
HF_AX_TINY_URL=$HF_ARGMAX_URL/openai_whisper-tiny
HF_AX_BASE_EN_URL=$HF_ARGMAX_URL/openai_whisper-base.en
HF_AX_BASE_URL=$HF_ARGMAX_URL/openai_whisper-base

QCOM_MODELS=(
    "$HF_QCOM_TINY_EN_URL"
    "$HF_QCOM_BASE_EN_URL"
    "$HF_QCOM_SMALL_EN_URL"
)

ARGMAX_MODELS=(
    "$HF_AX_TINY_EN_URL"
    "$HF_AX_TINY_URL"
    "$HF_AX_BASE_EN_URL"
    "$HF_AX_BASE_URL"
)

ARGMAX_MODEL_DIRECTORIES=(
  "$TINY_EN_MODELS_DIR"
  "$TINY_MODELS_DIR"
  "$BASE_EN_MODELS_DIR"
  "$BASE_MODELS_DIR"
)


QCOM_MODEL_DIRECTORIES=(
  "$QCOM_TINY_EN_MODELS_DIR"
  "$QCOM_BASE_EN_MODELS_DIR"
  "$QCOM_SMALL_EN_MODELS_DIR"
)

MODEL_DIRECTORIES=(
  "$QCOM_TINY_EN_MODELS_DIR"
  "$QCOM_BASE_EN_MODELS_DIR"
  "$QCOM_SMALL_EN_MODELS_DIR"
  "$TINY_MODELS_DIR"
  "$BASE_MODELS_DIR"
  "$TINY_EN_MODELS_DIR"
  "$BASE_EN_MODELS_DIR"
)

TOKENIZER_ENDPOINTS=(
  "$HF_OPENAI_TINY_EN_URL"
  "$HF_OPENAI_BASE_EN_URL"
  "$HF_OPENAI_SMALL_EN_URL"
  "$HF_OPENAI_TINY_URL"
  "$HF_OPENAI_BASE_URL"
  "$HF_OPENAI_TINY_EN_URL"
  "$HF_OPENAI_BASE_EN_URL"
)


ARIA2_OPTS="--quiet=true --summary-interval=0 --download-result=hide --continue=true --max-connection-per-server=4"

melspec_endpoint="${HF_ARGMAX_URL}/openai_whisper-tiny/MelSpectrogram.tflite"
aria2c $ARIA2_OPTS -d "/tmp/" -o MelSpectrogram.tflite $melspec_endpoint

for i in "${!MODEL_DIRECTORIES[@]}"; do
  model_dir="${MODEL_DIRECTORIES[$i]}"
  tokenizer_endpoint="${TOKENIZER_ENDPOINTS[$i]}"
  echo "Creating directory: $model_dir"
  mkdir -p "$model_dir"

  echo "Downloading tokenizer.json and config.json from $tokenizer_endpoint to $model_dir"

  aria2c $ARIA2_OPTS -d "$model_dir" -o tokenizer.json "$tokenizer_endpoint/tokenizer.json"
  aria2c $ARIA2_OPTS -d "$model_dir" -o config.json "$tokenizer_endpoint/config.json"
  echo "Done with $model_dir"
done

# Qualcomm models: rename to [AudioEncoder.tflite, TextDecoder.tflite]
#
# Argmax models: already named as [AudioEncoder.tflite, TextDecoder.tflite]

for i in "${!QCOM_MODEL_DIRECTORIES[@]}"; do
   model_dir="${QCOM_MODEL_DIRECTORIES[$i]}"
   model_endpoint="${QCOM_MODELS[$i]}"
   echo "Downloading QCOM [AudioEncoder, TextDecoder] from $model_endpoint to $model_dir"
   echo "Downloading Encoder: ${model_endpoint}/WhisperEncoder.tflite"
   echo "Downloading Decoder: ${model_endpoint}/WhisperDecoder.tflite"
   aria2c $ARIA2_OPTS -d "$model_dir" -o TextDecoder.tflite $model_endpoint/WhisperDecoder.tflite
   aria2c $ARIA2_OPTS -d "$model_dir" -o AudioEncoder.tflite $model_endpoint/WhisperEncoder.tflite
   cp /tmp/MelSpectrogram.tflite "$model_dir/MelSpectrogram.tflite"
   echo "Done with $model_dir"
done

for i in "${!ARGMAX_MODEL_DIRECTORIES[@]}"; do
   model_dir="${ARGMAX_MODEL_DIRECTORIES[$i]}"
   model_endpoint="${ARGMAX_MODELS[$i]}"
   echo "Downloading Argmax [AudioEncoder, TextDecoder] from $model_endpoint to $model_dir"
   echo "Downloading Encoder: ${model_endpoint}/AudioEncoder.tflite"
   echo "Downloading Decoder: ${model_endpoint}/TextDecoder.tflite"
   echo "Downloading MelSpec: ${model_endpoint}/MelSpectrogram.tflite"
   aria2c $ARIA2_OPTS -d "$model_dir" -o TextDecoder.tflite ${model_endpoint}/TextDecoder.tflite
   aria2c $ARIA2_OPTS -d "$model_dir" -o AudioEncoder.tflite ${model_endpoint}/AudioEncoder.tflite
   aria2c $ARIA2_OPTS -d "$model_dir" -o MelSpectrogram.tflite ${model_endpoint}/MelSpectrogram.tflite
   echo "Done with $model_dir"
done
