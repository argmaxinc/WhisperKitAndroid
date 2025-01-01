#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2023 Argmax, Inc. All Rights Reserved.
#

import os
import json
import unittest
import argparse
import subprocess
import concurrent.futures as futures

from pathlib import Path
from huggingface_hub import hf_hub_download
from whisper.tokenizer import get_tokenizer
from android_test_utils import AndroidTestsMixin
from linux_test_utils import LinuxTestsMixin


class TestWhisperKitAndroid(AndroidTestsMixin):
    TEST_DATASET = "librispeech-10mins"

    @classmethod
    def setUpClass(self):
        super().setUpClass()

    def __init__(self, methodName='runTest', args=None):
        super().__init__(methodName)
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

        metadata_path = os.path.join(self.path, "metadata.json")
        metadata_file = open(metadata_path)
        self.metadata = json.load(metadata_file)
        metadata_text = json.dumps(self.metadata)
        metadata_file.close()

        if os.path.isdir(self.args.input):
            for file in os.listdir(self.args.input):
                full_file_name = os.path.join(self.args.input, file)
                # checking if it is a file
                if os.path.isfile(full_file_name) is False:
                    continue
                
                name_only, extension = os.path.splitext(file)
                if extension not in self.audio_file_ext:
                    continue

                if metadata_text.find(name_only) == -1:
                    continue

                self.files.append(file)

    def test_audio_transcription(self):
        os.makedirs(self.args.output, exist_ok=True)
            
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
                    with open(f"{self.args.output}/{device}_report.json", "w")\
                        as json_file:
                        json.dump(output_json, json_file)


class TestWhisperKitLinux(LinuxTestsMixin):
    @classmethod
    def setUpClass(self):
        super().setUpClass()

    def __init__(self, methodName='runTest', args=None):
        super().__init__(methodName)
        self.args = args
        self.test_bin = "whisperax_cli"
        self.tokenizer = get_tokenizer(multilingual=False, language="en")

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

        metadata_path = os.path.join(self.path, "metadata.json")
        metadata_file = open(metadata_path)
        self.metadata = json.load(metadata_file)
        metadata_text = json.dumps(self.metadata)
        metadata_file.close()

        if os.path.isdir(self.args.input):
            for file in os.listdir(self.args.input):
                full_file_name = os.path.join(self.args.input, file)
                # checking if it is a file
                if os.path.isfile(full_file_name) is False:
                    continue
                
                name_only, extension = os.path.splitext(file)
                if extension not in self.audio_file_ext:
                    continue

                if metadata_text.find(name_only) == -1:
                    continue

                self.files.append(file)

    def test_audio_transcription(self):
        os.makedirs(self.args.output, exist_ok=True)

        output_json = self.run_test()
        if output_json is None:
            print('test on host has failed..')
            self.assertTrue(False)
        with open(f"{self.args.output}/linux_report.json", "w")\
            as json_file:
            json.dump(output_json, json_file)


def download_hg_dataset():
    test_path = f"{os.path.dirname(os.path.abspath(__file__))}/../test/"
    prefix = f"{test_path}/{TestWhisperKitAndroid.TEST_DATASET}"
    if os.path.exists(f"{prefix}/metadata.json"):
        return prefix
    
    hf_hub_download(
                "argmaxinc/whisperkit-test-data",
                filename="metadata.json",
                subfolder=TestWhisperKitAndroid.TEST_DATASET,
                local_dir=test_path,
                repo_type="dataset",
                revision="main",
    )
    metadata_file = open(os.path.join(prefix, "metadata.json"))
    metadata = json.load(metadata_file)
    metadata_file.close()
    
    file_name_filter = ["61-", "121-", "4507-"]
    for item in metadata:
        if "audio" not in item:
            continue
        if not any([item["audio"].startswith(f) for f in file_name_filter]):
            continue

        file = item["audio"].split(".")[0] + ".mp3"
        if os.path.exists(f"{prefix}/{file}"):
            continue
        try:
            hf_hub_download(
                "argmaxinc/whisperkit-test-data",
                filename=file,
                subfolder=TestWhisperKitAndroid.TEST_DATASET,
                local_dir=test_path,
                repo_type="dataset",
                revision="main",
            )
            print(f'downloaded {file}')
        except:
            print(' ') # this is not an issue, do nothing
        
    return prefix


class ArgParser(argparse.ArgumentParser):
    def print_help(self, file = None):
        super().print_help(file)
        print("example: ")
        print("  python3.10 test/whisperkit_test.py -m ./openai_whisper-tiny")


if __name__ == "__main__":
    parser = ArgParser()
    parser.add_argument("-m", "--model-path", 
            default="openai_whisper-tiny", type=str,
            help='path to the models, e.g., openai_whisper-tiny')
    parser.add_argument("-i", "--input",
            default=None, type=str,
            help='input file, folder, or (default) download from HuggingFace')
    parser.add_argument("-o", "--output",
            default="./output", type=str,
            help='output json folder')
    parser.add_argument("-l", "--linux", 
            action='store_true', 
            help='linux or (default) android device(s)')
    args = parser.parse_args()

    if args.input is None:
        args.input = download_hg_dataset()

    test_loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    runner = unittest.TextTestRunner()

    if args.linux:
        test_names = test_loader.getTestCaseNames(TestWhisperKitLinux)
        for test_name in test_names:
            suite.addTest(TestWhisperKitLinux(test_name, args=args))
    else:
        test_names = test_loader.getTestCaseNames(TestWhisperKitAndroid)
        for test_name in test_names:
            suite.addTest(TestWhisperKitAndroid(test_name, args=args))

    allsuites = unittest.TestSuite([suite])
    runner.run(allsuites)