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
from android_test_utils import AndroidTestsMixin, TestRunADB

OUTPUT_FOLDER = './json/'
TEST_DATASET = "librispeech-10mins" 
TEST_FILES = [
    "121-121726-0000.mp3", "121-121726-0001.mp3", "121-121726-0002.mp3", 
    "121-121726-0003.mp3", "121-121726-0004.mp3", "121-121726-0005.mp3", 
    "121-121726-0006.mp3", "121-121726-0007.mp3", "121-121726-0010.mp3", 
    "121-121726-0011.mp3", "121-121726-0012.mp3", "121-121726-0013.mp3", 
    "121-121726-0014.mp3", "121-123852-0000.mp3", "121-123852-0001.mp3", 
    "121-123852-0002.mp3", "121-123852-0003.mp3", "121-123852-0004.mp3", 
    "121-123859-0000.mp3", "121-123859-0001.mp3", "121-123859-0002.mp3", 
    "121-123859-0003.mp3", "121-123859-0004.mp3", "4507-16021-0000.mp3", 
    "4507-16021-0001.mp3", "4507-16021-0002.mp3", "4507-16021-0003.mp3", 
    "61-70968-0000.mp3", "61-70968-0001.mp3", "61-70968-0002.mp3", 
    "61-70968-0003.mp3", "61-70968-0004.mp3", "61-70968-0005.mp3", 
    "61-70968-0006.mp3", "61-70968-0007.mp3", "61-70968-0008.mp3", 
    "61-70968-0009.mp3", "61-70968-0010.mp3", "61-70968-0011.mp3", 
    "61-70968-0012.mp3", "61-70968-0013.mp3", "61-70968-0014.mp3", 
    "61-70968-0015.mp3", "61-70968-0016.mp3", "61-70968-0017.mp3", 
    "61-70968-0018.mp3", "61-70968-0019.mp3", "61-70968-0020.mp3", 
    "61-70968-0021.mp3", "61-70968-0022.mp3", "61-70968-0023.mp3", 
    "61-70968-0024.mp3", "61-70968-0025.mp3", "61-70968-0026.mp3", 
    "61-70968-0027.mp3", "61-70968-0028.mp3", "61-70968-0029.mp3", 
    "61-70968-0030.mp3", "61-70968-0031.mp3", "61-70968-0032.mp3", 
    "61-70968-0033.mp3", "61-70968-0034.mp3", "61-70968-0035.mp3", 
    "61-70968-0036.mp3", "61-70968-0037.mp3", "61-70968-0038.mp3", 
    "61-70968-0039.mp3", "61-70968-0040.mp3", "61-70968-0041.mp3", 
    "61-70968-0042.mp3", "61-70968-0043.mp3", "61-70968-0044.mp3", 
    "61-70968-0045.mp3", "61-70968-0046.mp3", "61-70968-0047.mp3", 
    "61-70968-0048.mp3", "61-70968-0049.mp3", "61-70968-0050.mp3", 
    "61-70968-0051.mp3", "61-70968-0052.mp3", "61-70968-0053.mp3", 
    "61-70968-0054.mp3", "61-70968-0055.mp3", "61-70968-0056.mp3", 
    "61-70968-0057.mp3", "61-70968-0061.mp3", "61-70968-0062.mp3", 
    "61-70970-0000.mp3", "61-70970-0001.mp3", "61-70970-0002.mp3", 
    "61-70970-0003.mp3", "61-70970-0004.mp3", "61-70970-0005.mp3", 
    "61-70970-0006.mp3", "61-70970-0007.mp3", "61-70970-0008.mp3", 
    "61-70970-0009.mp3", "61-70970-0010.mp3", "61-70970-0011.mp3", 
    "61-70970-0012.mp3", "61-70970-0013.mp3"
]
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


def download_hg_dataset():
    test_path = os.path.dirname(os.path.abspath(__file__)) + "/../test/"
    hf_hub_download(
                "argmaxinc/whisperkit-test-data",
                filename="metadata.json",
                subfolder=TEST_DATASET,
                local_dir=test_path,
                repo_type="dataset",
                revision="main",
    )

    for file in TEST_FILES:
        if os.path.exists(test_path + "/" + TEST_DATASET + "/" + file):
            continue
        try:
            hf_hub_download(
                "argmaxinc/whisperkit-test-data",
                filename=file,
                subfolder=TEST_DATASET,
                local_dir=test_path,
                repo_type="dataset",
                revision="main",
            )
            print(f'downloaded {file}')
        except:
            print(f'can not find {file} from Huggingface')
        
    return test_path + "/" + TEST_DATASET


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", 
            default="tiny", type=str, 
            choices=("tiny", "base", "small"))
    parser.add_argument("-i", "--input", 
            default=None, type=str, 
            help='input file, folder, or (default) download from HuggingFace')
    args = parser.parse_args()

    if args.input is None:
        args.input = download_hg_dataset()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWhisperKitAndroid)
    allsuites = unittest.TestSuite([suite])
    runner = unittest.TextTestRunner()
    runner.run(allsuites)