#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import unittest
import subprocess
import os
import time
import json
import evaluate
import statistics
import re
from threading import Thread, Condition, Lock
from whisper.normalizers import EnglishTextNormalizer
from difflib import Differ
from collections import defaultdict


class TestRunADB:

    def __init__(
        self,
        config,
        root_path,
        tokenizer,
        serial
    ):
        """
        Args:
            bin_path, lib_path:
                Path under /data/local/tmp for executorch binary & library files
            root_path:   Path under /sdcard/argmax, for other audio/model files
            serial:      device serial no.
        """
        self.bin_path = "/data/local/tmp/bin"
        self.lib_path = "/data/local/tmp/lib"
        self.root_path = root_path
        self.config = config
        self.executing = False
        self.condition = Condition()
        self.batts = []
        self.temps = []
        self.mem = 0
        self.curr_cmd = None
        self.output_json = None
        self.probe_thread = None
        self.tokenizer = tokenizer
        self.text_normalizer = EnglishTextNormalizer()
        self.serial = serial
        self.delegate_file = None
        self.first_exec = True
        self.wers = []
        self.total_tokens = 0
        self.sum_time_elapsed = 0.0

    def _adb(self, cmd):
        cmds = ["adb", "-s", self.serial]
        cmds.extend(cmd)
        output = subprocess.run(cmds, stdout=subprocess.PIPE)
        if output.stdout is None:
            return None
        return output.stdout.decode('utf-8')

    def _check_device(self):
        result = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE)
        output = result.stdout.decode()
        if "device" in output:
            if self.serial in output:
                return True
 
        print(f"ADB Device {self.serial} not found")
        return False

    def push_file(self, file, subfolder='.'):
        dest_folder = f"{self.root_path}/{subfolder}/"
        _ = self._adb(["push", file, dest_folder])

    def device_test(self, test_bin, input_audio, model_path):
        if self._check_device() is False:
            return 
        if test_bin is None: 
            return 

        input_audio = f"{self.root_path}/inputs/{input_audio}"
        self.curr_cmd = test_bin
        test_cmds = " ".join(
            [
                f"cd {self.root_path} &&",
                f"export LD_LIBRARY_PATH={self.lib_path} &&",
                f"{self.bin_path}/{self.curr_cmd} ",
                f"--audio-path {input_audio} ",
                f"--model-path {model_path} ",
                f"--report --report-path ."
            ]
        )

        if self.first_exec:
            test_cmds += f" --verbose > {self.config['test']['delegate_file']}"

        print(f"Running: {test_cmds}")
        self.executing = True
        with self.condition:
            self.condition.notify()

        self.run_shell_cmds(f"{test_cmds}")
        self.executing = False

    def pull(self, file, callback=None):
        device_path = f"{self.root_path}/{file}"
        host_path = f"{self.serial}_{file}"
        _ = self._adb(["pull", device_path, host_path])
        if callback:
            callback()
        return host_path

    def run_shell_cmds(self, command):
        return self._adb(["shell", command])

    def get_meminfo(self, test_binary):
        strout = self.run_shell_cmds(f'dumpsys meminfo {test_binary}')
        for line in strout.split('\n'):
            if 'TOTAL RSS:' in line:
                items = line.split()
                if int(items[5]) > self.mem:
                    self.mem = int(items[5])
                break
        return self.mem

    def get_batterylevel(self):
        strout = self.run_shell_cmds('dumpsys battery | grep level')
        items = strout.split()

        idx = 0
        try:
            idx = items.index("level:")
        except:
            return -1
        
        return int(items[idx+1])

    def get_thermalinfo(self):
        highest_temp = 0.0
        strout = self.run_shell_cmds('dumpsys thermalservice')
        for line in strout.split('\n'):
            if 'mName=AP' in line or 'mName=CPU' in line:
                substr = line.split('=')
                substr = substr[1].split(',')
                temp = float(substr[0])
                if highest_temp < temp:
                    highest_temp = temp
        return highest_temp

    def _probe_device(self, test_binary):
        self.batts.append(self.get_batterylevel())
        self.temps.append(self.get_thermalinfo())

        if self.executing is False:
            with self.condition:
                self.condition.wait()

        while self.executing:
            self.get_meminfo(test_binary)
            time.sleep(0.2)

        self.batts.append(self.get_batterylevel())
        self.temps.append(self.get_thermalinfo())

    def _start_probe(self, test_binary):
        self.probe_thread = Thread(
                            target=self._probe_device, 
                            args=(test_binary,))
        self.probe_thread.start()

    def _end_probe(self):
        if self.probe_thread is None:
            return 
        self.probe_thread.join()

    def _get_wer(self, ref, pred):
        wer_metric = evaluate.load("argmaxinc/detailed-wer")
        detailed_wer = wer_metric.compute(
            references=[ref,],
            predictions=[pred,],
            detailed=True
        )
        return detailed_wer
    
    def _get_text_diffs(self, ref, pred):
        d = Differ()
        reference_words = ref.split()
        prediction_words = pred.split()
        
        diffs = []
        for token in d.compare(reference_words, prediction_words):
            if token.startswith('?'):
                continue
                
            status = token[0]
            word = token[2:].strip()
            
            if word:  
                if status == ' ':
                    diffs.append((word, None))  
                else:
                    diffs.append((word, status))  
                    
        return diffs
    
    def _parse_delegate_file(self, file_path):
        delegate_counts = defaultdict(int)
        total_nodes = 0
        
        pattern = r"Node \d+ is a (TfLite\w+) node .+?, which has delegated (\d+) nodes:"
        
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    delegate_type = match.group(1)
                    node_count = int(match.group(2))
                    delegate_counts[delegate_type] += node_count
                    total_nodes += node_count
        
        for delegate, count in delegate_counts.items():
            percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
            delegate_counts[delegate] = (count, f"{percentage}%")

        return delegate_counts        

    def _put_test_info(self, data_set, file, metadata):
        reference = None
        name_only, _ = os.path.splitext(file)
        for data in metadata:
            if str(data['audio']).find(name_only) != -1:
                reference = data['text']
                break

        device_info = {}
        device_info["peak_mem"] = self.mem
        device_info["cpu_temp"] = self.temps[1]
        device_info["batt_level"] = self.batts[1]

        if self.delegate_file and os.path.exists(self.delegate_file):
            delegation_info = self._parse_delegate_file(self.delegate_file)
            device_info["delegationInfo"] = delegation_info

        self.output_json["deviceInfo"] = json.dumps(device_info)
        self.output_json["testInfo"]["datasetDir"] = data_set
        self.output_json["testInfo"]["reference"] = reference
        
        prediction_token = self.output_json["testInfo"]["prediction"]
        prediction_text = self.tokenizer.decode(prediction_token)
        normalized = self.text_normalizer(prediction_text)
        wer = self._get_wer(reference, normalized)
        
        self.output_json["testInfo"]["prediction"] = normalized
        self.output_json["testInfo"]["diff"] = self._get_text_diffs(reference, normalized)
        self.output_json["testInfo"]["wer"] = wer["wer"]
        self.output_json["testInfo"]["substitutionRate"] = wer["substitution_rate"]
        self.output_json["testInfo"]["deletionRate"] = wer["deletion_rate"]
        self.output_json["testInfo"]["insertionRate"] = wer["insertion_rate"]
        self.output_json["testInfo"]["numSubstitutions"] = wer["num_substitutions"]
        self.output_json["testInfo"]["numDeletions"] = wer["num_deletions"]
        self.output_json["testInfo"]["numInsertions"] = wer["num_insertions"]
        self.output_json["testInfo"]["numHits"] = wer["num_hits"]

    def run_test(
            self, test_binary, 
            file, data_set, 
            metadata, model_size):
        self._start_probe(test_binary)
        result = self.device_test(test_binary, file, model_size)
        self._end_probe()

        if result is False:
            return None
        
        output_file = self.pull(file=self.config['test']['output_file'])

        if os.path.exists(output_file) is False:
            return None

        if self.first_exec:
            self.delegate_file = self.pull(file=self.config['test']['delegate_file'])
            self.first_exec = False

        f = open(output_file)
        self.output_json = json.load(f)
        f.close()
        os.remove(output_file)

        self._put_test_info(data_set, file, metadata)
        self.total_tokens += self.output_json["latencyStats"]["measurements"]["cumulativeTokens"]
        self.sum_time_elapsed += self.output_json["latencyStats"]["measurements"]["timeElapsed"]

        print(f" ** avg WER: {statistics.mean(self.wers)}")
        print(f" ** toks/sec: {self.total_tokens / self.sum_time_elapsed}")
        return self.output_json


class AndroidTestsMixin(unittest.TestCase):
    """ Mixin class for Android device test with audio input file
    """
    @classmethod
    def setUpClass(self):
        self.test_no = 0
        self.lock = Lock()

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        test_path = f"{os.path.dirname(os.path.abspath(__file__))}"
        config_file = open(os.path.join(test_path, "ENVIRONMENT.json"))
        self.config = json.load(config_file)
        config_file.close()
        self.audio_file_ext = self.config['audio']['extensions']
        self.test_path = f"{test_path}/dataset/{self.config['test']['datasets'][0]}"

    def run_test(self, device):
        adb = TestRunADB(self.config, self.root_path, 
                         self.tokenizer, device)
    
        outputs_json = []
        idx = 0
        intermediate_file = f"{self.args.output}/{device}_intermediate.json"
        for file in self.files:
            with self.lock:
                test_no = self.test_no
                self.test_no += 1

            full_path = os.path.join(self.test_path, file)
            adb.push_file(full_path, self.config['audio']['local_dir'])

            print(f'======== Running test #{test_no} (audio: {file}) on {device} ========')
            
            output = adb.run_test(
                self.test_bin, file, self.data_set, 
                self.metadata, self.args.model_path)
            
            print(f'======== Completed test #{test_no} (audio: {file}) on {device} ========')

            outputs_json.append(output)
            
            idx += 1
            if idx % 10 == 0:
                with open(intermediate_file, "w") as json_file:
                    json.dump(outputs_json, json_file)
            time.sleep(1)

        json.dumps(outputs_json)

        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)
        
        return adb.delegate_file, outputs_json
