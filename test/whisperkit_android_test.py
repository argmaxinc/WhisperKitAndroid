#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import os
import json
import unittest
import argparse
import subprocess
import concurrent.futures as futures

from pathlib import Path
from whisper.tokenizer import get_tokenizer
from android_test_utils import AndroidTestsMixin, TestRunADB

OUTPUT_FOLDER = './json/'
args = None

class TestWhisperKitAndroid(AndroidTestsMixin):
    @classmethod
    def setUpClass(self):
        super().setUpClass()
        self.args = args
        self.test_bin = "whisperax_cli"
        self.root_path = "/sdcard/argmax/tflite"
        self.devices = []
        self.tokenizer = get_tokenizer(multilingual=False, language="en")

        output = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE)
        if output.stdout is None:
            return None
        devices = output.stdout.decode('utf-8')
        words = devices.split()

        attached_found = False
        for index, word in enumerate(words):
            if attached_found is False:
                if word.find('attached') != -1:
                    attached_found = True
                else:
                    continue

            if len(words) > index + 1:
                if words[index + 1].find("device") != -1:
                    self.devices.append(word)

        splitted = os.path.split(os.path.abspath(self.args.input))
        path_parts = Path(self.args.input).parts
        if os.path.isfile(self.args.input):
            self.path = splitted[0]
            self.files = [splitted[1]]
            self.data_set = path_parts[-2]
        
        elif os.path.isdir(self.args.input):
            self.path = self.args.input
            self.data_set = path_parts[-1]
            self.files = []

        if os.path.isdir(self.args.input):
            for file in os.listdir(self.args.input):
                full_file_name = os.path.join(self.args.input, file)
                # checking if it is a file
                if os.path.isfile(full_file_name) is False:
                    continue
                
                _, extension = os.path.splitext(file)
                if extension not in self.audio_file_ext:
                    continue

                self.files.append(file)

    def test_audio_transcription(self):
        if os.path.exists(OUTPUT_FOLDER) is False:
            os.mkdir(OUTPUT_FOLDER)
            
        with futures.ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            future_test = { 
                executor.submit(self.run_test, device):
                    device for device in self.devices
            }

            for future in futures.as_completed(future_test):
                device = future_test[future]
                try: 
                    output_json = future.result()
                except Exception as exc:
                    print('device %r generated an exception: %s' % (device, exc))
                else:
                    if output_json is None:
                        print('device %r test has failed..')
                        self.assertTrue(False)
                    with open(OUTPUT_FOLDER + device + "_" + TestRunADB.output_file, "w")\
                        as json_file:
                        json.dump(output_json, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", default="tiny", type=str, choices=("tiny", "base", "small"))
    parser.add_argument("-i", "--input", default=None, type=str)
    args = parser.parse_args()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWhisperKitAndroid)
    allsuites = unittest.TestSuite([suite])
    runner = unittest.TextTestRunner()
    runner.run(allsuites)