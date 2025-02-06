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
import statistics
import docker
import threading
from whisper.normalizers import EnglishTextNormalizer


# A class for running tests on Linux host.
# Any commands on docker is to be run via _run_docker_cmd()
class TestRunLinux:

    def __init__(
        self,
        config,
        tokenizer
    ):
        self.config = config
        self.root = os.path.dirname(os.path.abspath(__file__)) + "/.."
        self.bin_path = self.config['docker']['rootdir'] + "/build/linux"
        self.lib_path = self.config['docker']['rootdir'] + "/external/libs/linux"
        self.output_json = {}
        self.report_json = {}
        self.tokenizer = tokenizer
        self.text_normalizer = EnglishTextNormalizer()
        self.container = docker.from_env()\
                            .containers.get(self.config['docker']['container'])
        self.wers = []
        self.total_tokens = 0
        self.sum_time_elapsed = 0.0

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

    def copy_file(self, file, subfolder='.'):
        dest_folder =  f"{self.root}/{subfolder}/"
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        shutil.copy(file, dest_folder)

    def device_test(self, test_bin, input_audio, model_path):
        if test_bin is None: 
            return False

        test_cmds = " ".join(
            [
                f"export LD_LIBRARY_PATH={self.lib_path} &&",
                f"{self.bin_path}/{test_bin} ",
                f"--audio-path {input_audio} ",
                f"--model-path {model_path} ",
                f"--report --report-path ."
            ]
        )
        print(f"Running: {test_cmds}")
        self._run_docker_cmd(test_cmds)
        return True

    def _get_wer(self, ref, pred):
        wer_metric = evaluate.load("argmaxinc/detailed-wer")
        detailed_wer = wer_metric.compute(
            references=[ref,],
            predictions=[pred,],
            detailed=True
        )
        return detailed_wer

    def _put_test_info(self, data_set, file, metadata):
        reference = None
        name_only, _ = os.path.splitext(file)
        for data in metadata:
            if str(data['audio']).find(name_only) != -1:
                reference = data['text']
                break

        prediction_token = self.output_json["testInfo"]["prediction"]
        prediction_text = self.tokenizer.decode(prediction_token)
        normalized = self.text_normalizer(prediction_text)
        
        file = self.output_json["testInfo"]["audioFile"]
        audio_duration = self.output_json["testInfo"]["timings"]["inputAudioSeconds"]
        wer = self._get_wer(reference, normalized)

        self.wers.append(wer["wer"])
        
        self.report_json = {}
        self.report_json["reference"] = reference
        self.report_json["prediction"] = normalized
        self.report_json["wer"] = wer["wer"]
        self.report_json["file"] = file
        self.report_json["audioDuration"] = audio_duration
        self.report_json["substitutionRate"] = wer["substitution_rate"]
        self.report_json["deletionRate"] = wer["deletion_rate"]
        self.report_json["insertionRate"] = wer["insertion_rate"]
        self.report_json["numSubstitutions"] = wer["num_substitutions"]
        self.report_json["numDeletions"] = wer["num_deletions"]
        self.report_json["numInsertions"] = wer["num_insertions"]
        self.report_json["numHits"] = wer["num_hits"]

    def run_test(
            self, test_binary, 
            file, data_set, 
            metadata, model_path):
        rootdir = self.config['docker']['rootdir']
        localdir = self.config['audio']['local_dir']
        full_path = f"{rootdir}/{localdir}/{file}"
        result = self.device_test(test_binary, full_path, model_path)
        if result is False:
            return None
        
        output_file = f"{self.root}/{self.config['test']['output_file']}"
        if os.path.exists(output_file) is False:
            return None

        f = open(output_file)
        self.output_json = json.load(f)
        f.close()
        os.remove(output_file)

        self._put_test_info(data_set, file, metadata)
        self.total_tokens += self.output_json["latencyStats"]["measurements"]["cumulativeTokens"]
        self.sum_time_elapsed += self.output_json["latencyStats"]["measurements"]["timeElapsed"]

        print(f" ** avg WER: {statistics.mean(self.wers)}")
        print(f" ** toks/sec: {self.total_tokens / self.sum_time_elapsed}")        
        return self.report_json


class LinuxTestsMixin(unittest.TestCase):
    """ Mixin class for Linux test with audio input file
    """

    @classmethod
    def setUpClass(self):
        self.test_no = 0
        self.lock = threading.Lock()

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        test_path = f"{os.path.dirname(os.path.abspath(__file__))}"
        config_file = open(os.path.join(test_path, "ENVIRONMENT.json"))
        self.config = json.load(config_file)
        config_file.close()
        self.audio_file_ext = self.config['audio']['extensions']
        self.test_path = f"{test_path}/dataset/{self.config['test']['datasets'][0]}"

    def run_test(self):
        host = TestRunLinux(self.config, self.tokenizer)
    
        outputs_json = {"results": []}
        for file in self.files:
            with self.lock:
                test_no = self.test_no
                self.test_no += 1

            full_path = os.path.join(self.test_path, file)
            host.copy_file(full_path, self.config['audio']['local_dir'])

            print(f'======== Running test #{test_no} (audio: {file}) on linux host ========')
            
            output = host.run_test(
                self.test_bin, file, 
                self.data_set, self.metadata, 
                self.args.model_path)
            
            print(f'======== Completed test #{test_no} (audio: {file}) on linux host ========')

            outputs_json["results"].append(output)
            time.sleep(1)

        json.dumps(outputs_json)
        return outputs_json
