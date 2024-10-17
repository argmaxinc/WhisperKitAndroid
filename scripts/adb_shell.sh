#!/usr/bin/expect --
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

spawn adb shell
expect "$" {
    sleep 0.1
    send "export LD_LIBRARY_PATH=/data/local/tmp/lib; export ADSP_LIBRARY_PATH=/data/local/tmp/lib; export PATH=/data/local/tmp/bin:\$PATH; axie_tflite \n"
}
interact
