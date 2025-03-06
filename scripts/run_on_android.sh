#!/system/bin/sh

# Set up environment variables for the Android session
export PATH=/data/local/tmp/bin:$PATH
export LD_LIBRARY_PATH=/data/local/tmp/lib

whisperkit-cli transcribe --model-path /sdcard/argmax/tflite/models/openai_whisper-tiny --audio-path /sdcard/argmax/tflite/inputs/jfk_441khz.m4a
