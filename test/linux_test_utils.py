#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.
#

import unittest
import shutil
import os
import time
import json
import evaluate
import docker
from whisper.normalizers import EnglishTextNormalizer


class TestRunLinux:
    """
    A class for running tests on Linux host.
    Any commands on docker is to be run via _run_docker_cmd()
    """
    output_file = "output.json"
    container_name = "axie_tflite"
    docker_root = "/src/AXIE"
    
    def __init__(
        self,
        tokenizer
    ):
        self.root = os.path.dirname(os.path.abspath(__file__)) + "/.."
        self.bin_path = self.docker_root + "/build/linux"
        self.lib_path = self.docker_root + "/external/libs/linux"
        self.output_json = None
        self.tokenizer = tokenizer
        self.text_normalizer = EnglishTextNormalizer()
        self.docker = docker.from_env()
        self.container = self.docker.containers.get(self.container_name)

    def _run_docker_cmd(self, cmd):
        cmds = f"/bin/bash -c '{cmd}'"
        try:
            _, result = self.container.exec_run(cmds, stream=True)
        except Exception as e:
            print(f"** Error from running a command on Docker container:")
            print(f"  Is {self.config['docker']['container']} running?")
            print(f"  If not, run 'make env' to start Docker container")
            os._exit(-1)
        
        for log in result:
            line = log.decode()
            print(line)
            find_duration = "Total Duration:"
            pos = line.find(find_duration)
            if pos > 0:
                strs = line[(pos + len(find_duration) + 1): -1].split()
                return float(strs[0])
        return 0.0

    def copy_file(self, file, subfolder='.'):
        dest_folder =  f"{self.root}/{subfolder}/"
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        shutil.copy(file, dest_folder)

    def device_test(self, test_bin, input_audio, model_size):
        if test_bin is None: 
            return False

        test_cmds = " ".join(
            [
                f"export LD_LIBRARY_PATH={self.lib_path} &&",
                f"{self.bin_path}/{test_bin} ",
                f"{input_audio} {model_size} debug",
            ]
        )
        print(f"Running: {test_cmds}")
        duration = self._run_docker_cmd(test_cmds)
        if duration <= 0.0:
            print(f"An Error Occurred..")
            return False
        return True

    def _get_wer(self, ref, pred):
        wer_metric = evaluate.load("wer")
        avg_wer = wer_metric.compute(
            references=[ref,],
            predictions=[pred,],
        )
        avg_wer = round(100 * avg_wer, 2)
        return avg_wer

    def _put_test_info(self, data_set, file, metadata):
        reference = None
        name_only, _ = os.path.splitext(file)
        for data in metadata:
            if str(data['audio']).find(name_only) != -1:
                reference = data['text']
                break

        self.output_json["testInfo"]["datasetDir"] = data_set
        self.output_json["testInfo"]["reference"] = reference
        
        prediction_token = self.output_json["testInfo"]["prediction"]
        prediction_text = self.tokenizer.decode(prediction_token)
        normalized = self.text_normalizer(prediction_text)
        self.output_json["testInfo"]["prediction"] = normalized
        self.output_json["testInfo"]["wer"] = self._get_wer(reference, normalized)

    def run_test(
            self, test_binary, 
            file, data_set, 
            metadata, model_size):
        full_path = f"{self.docker_root}/inputs/{file}"
        result = self.device_test(test_binary, full_path, model_size)
        if result is False:
            return None
        
        output_file = f"{self.root}/{TestRunLinux.output_file}"
        if os.path.exists(output_file) is False:
            return None

        f = open(output_file)
        self.output_json = json.load(f)
        f.close()
        os.remove(output_file)

        self._put_test_info(data_set, file, metadata)
        return self.output_json


class LinuxTestsMixin(unittest.TestCase):
    """ Mixin class for Linux test with audio input file
    """
    audio_file_ext = [".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wav"]

    @classmethod
    def setUpClass(self):
        self.test_no = 0

    def run_test(self):
        self.test_no += 1

        if self.args.model_path.find("openai_whisper-tiny") != -1:
            model_size = "tiny"
        elif self.args.model_path.find("openai_whisper-base") != -1:
            model_size = "base"
        elif self.args.model_path.find("openai_whisper-small") != -1:
            model_size = "small"

        host = TestRunLinux(self.tokenizer)
    
        outputs_json = []
        for file in self.files:
            test_no = self.test_no
            full_path = os.path.join(self.path, file)
            host.copy_file(full_path, "inputs")

            print(f'======== Running test #{test_no} (audio: {file}) on linux host ========')
            
            output = host.run_test(
                self.test_bin, file, 
                self.data_set, self.metadata, 
                model_size)
            
            print(f'======== Completed test #{test_no} (audio: {file}) on linux host ========')

            self.test_no += 1
            outputs_json.append(output)
            time.sleep(1)

        json.dumps(outputs_json)
        return outputs_json
