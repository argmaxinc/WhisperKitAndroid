<div align="center">

<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# WhisperKit Android (Beta) Test
</div>

Test on Android device and build its output JSON file. 


## Table of Contents
- [Installation](#installation)
- [Test with file/folder](#test-with-file--folder)
- [Citation](#citation)

## Installation

<details>
  <summary> (Click to expand) </summary>

The following setup was tested on macOS 15.1 & with python 3.10.15.

1. Ensure you install the required python packages:
```
make test
```

The first time running `make test` command will take several minutes, by installing required packages in test/requirements.txt

</details>

## Test with file & folder

<details>
  <summary> (Click to expand) </summary>

1. Ensure that you have more than 1 Android device connected:
```
adb devices
```

2. Test with a single audio file :
```
cd test
python3 whisperkit_android_test.py -i ./data/jfk_441khz.m4a -m tiny
```

3. After the test is done, check the result JSON file:
```
 cat json/{device serial number}_output.json

[{"latencyStats": {"measurements": {"cumulativeTokens": 28, "numberOfMeasurements": 28, "timeElapsed": 4.479000091552734}, "totalNumberOfMeasurements": 28, "units": "Tokens/Sec"}, "staticAttributes": {"os": "Android 13"}, "testInfo": {"audioFile": "jfk_441khz.m4a", "date": "2024-12-05T19:56:44Z", "device": "Xiaomi MT6985", "model": "openai_whisper-tiny", "prediction": "and so my fellow american asked not what your country can do for you ask what you can do for your country", "timeElapsedInSeconds": 5.715000152587891, "timings": {"fullPipeline": 5.2910003662109375, "inputAudioSeconds": 11.0, "totalDecodingFallbacks": 0, "totalDecodingLoops": 28, "totalEncodingRuns": 1}, "datasetDir": "data", "reference": "None"}, "deviceInfo": "{\"peak_mem\": 723356, \"cpu_temp\": 54.481, \"batt_level\": 100}"}]
```

4. Test with multiple audio files in a folder:
```
python3 whisperkit_android_test.py -i ./data -m tiny
```

</details>

## Citation
If you use WhisperKit for something cool or just find it useful, please drop us a note at [info@argmaxinc.com](mailto:info@argmaxinc.com)!

If you are looking for managed enterprise deployment with Argmax, please drop us a note at [info+sales@argmaxinc.com](mailto:info+sales@argmaxinc.com).

If you use WhisperKit for academic work, here is the BibTeX:

```bibtex
@misc{whisperkit-argmax,
   title = {WhisperKit},
   author = {Argmax, Inc.},
   year = {2024},
   URL = {https://github.com/argmaxinc/WhisperKit}
}
```