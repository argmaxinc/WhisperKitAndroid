#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."

pip install -r $SOURCE_DIR/test/requirements.txt

python3 $SOURCE_DIR/test/whisperkit_android_test.py -i $SOURCE_DIR/test/data -m tiny 
# python3 $SOURCE_DIR/test/whisperkit_android_test.py -i $SOURCE_DIR/test/data -m base
# python3 $SOURCE_DIR/test/whisperkit_android_test.py -i $SOURCE_DIR/test/data -m small