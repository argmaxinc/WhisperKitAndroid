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
from threading import Thread, Condition
from whisperkit.evaluate.normalize_en import EnglishTextNormalizer


class TestRunADB:
    output_file = "output.json"

    def __init__(
        self,
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

    def _adb(self, cmd):
        cmds = ["adb", "-s", self.serial]
        cmds.extend(cmd)
        # subprocess.run(cmds, stdout=sys.stdout)
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
        dest_folder = self.root_path + "/" + subfolder + "/"
        _ = self._adb(["push", file, dest_folder])

    def device_test(self, test_bin, input_audio, model_size):
        if self._check_device() is False:
            return 
        if test_bin is None: 
            return 

        input_audio = self.root_path + "/inputs/" + input_audio
        self.curr_cmd = test_bin
        test_cmds = " ".join(
            [
                f"cd {self.root_path} &&",
                f"export LD_LIBRARY_PATH={self.lib_path} &&",
                f"{self.bin_path}/{self.curr_cmd} ",
                f"{input_audio} {model_size} debug",
            ]
        )
        print(f"Running: {test_cmds}")
        self.executing = True
        with self.condition:
            self.condition.notify()

        self.run_shell_cmds(f"{test_cmds}")
        self.executing = False

    def pull(self, file, callback=None):
        device_path = self.root_path + "/" + file
        host_path = self.serial + "_" + file
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

        device_info = {}
        device_info["peak_mem"] = self.mem
        device_info["cpu_temp"] = self.temps[1]
        device_info["batt_level"] = self.batts[1]

        self.output_json["deviceInfo"] = json.dumps(device_info)
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
        self._start_probe(test_binary)
        result = self.device_test(test_binary, file, model_size)
        self._end_probe()

        if result is False:
            return None
        
        output_file = self.pull(file=TestRunADB.output_file)

        if os.path.exists(output_file) is False:
            return None

        f = open(output_file)
        self.output_json = json.load(f)
        f.close()
        os.remove(output_file)

        self._put_test_info(data_set, file, metadata)
        return self.output_json


class AndroidTestsMixin(unittest.TestCase):
    """ Mixin class for Android device test with audio input file
    """
    audio_file_ext = [".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wav"]

    @classmethod
    def setUpClass(self):
        self.test_no = 0

    def run_test(self, device):
        self.test_no += 1
        adb = TestRunADB(self.root_path, self.tokenizer, device)
    
        outputs_json = []
        for file in self.files:
            test_no = self.test_no
            full_path = os.path.join(self.path, file)
            adb.push_file(full_path, "inputs")

            print(f'======== Running test #{test_no} (audio: {file}) on {device} ========')
            
            output = adb.run_test(
                self.test_bin, 
                file, self.data_set, 
                self.metadata, self.args.model_size)
            
            print(f'======== Completed test #{test_no} (audio: {file}) on {device} ========')

            self.test_no += 1
            outputs_json.append(output)
            time.sleep(1)

        json.dumps(outputs_json)
        return outputs_json
