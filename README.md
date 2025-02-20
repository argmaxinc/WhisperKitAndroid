<div align="center">

<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# WhisperKit Android (Beta)
</div>

WhisperKit Android brings Foundation Models On Device for Automatic Speech Recognition. It extends the performance and feature set of [WhisperKit](https://github.com/argmaxinc/WhisperKit) from Apple platforms to Android and Linux.  The current feature set is a subset of the iOS counterpart, 
but we are continuing to invest in Android and now welcome contributions from the community.

[Example App (Coming Soon)] [[Blog Post]](https://takeargmax.com/blog/android) [[Python Tools Repo]](https://github.com/argmaxinc/whisperkittools)


## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [CLI Run and Test](#cli-run-and-test)
- [Running Android app](#running-android-app)
- [Contributing \& Roadmap](#contributing--roadmap)
- [License](#license)
- [Citation](#citation)

# Installation

<details>
  <summary> (Click to expand) </summary>

The following setup was tested on macOS 15.1.

1. Ensure you have the required build tools using:
```
make setup
```

2. Download Whisper models (<1.5GB) and auxiliary files
```
make download-models
```

3. Build development environment in Docker with all development tools (~12GB):
```
make env
```

The first time running `make env` command will take several minutes.

After the Docker image builds, the next time running `make env` will execute inside the Docker container right away.

You can use the following to rebuild the Docker image, if needed:
```
make rebuild-env
```

</details>

# Getting Started

<details>
  <summary> (Click to expand) </summary>

WhisperKit Android is a Whisper pipeline built on top of Tensorflow Lite (LiteRT) with a provided 
CLI interface via `whisperkit-cli`.  The library is built with a C API for Android and Linux.  Please 
note that as the library is currently in Beta, the C API is not yet stable.

1. Execute into the Docker build environment:
```
make env
```

2. Inside the Docker environment, build the `whisperkit-cli` CLI using (for Android and Linux):
```
make build [linux | qnn | gpu | jni]
```

The QNN option builds WhisperKit with Qualcomm AI NPU support and the QNN TFLite delegate.
The 'gpu' option is the generic GPU backend for all Android devices from TFLite GPU delegate.
Linux builds are currently CPU-only.
The 'jni' option builds the .so file with JNI library to use on android (using QNN support).

3. Back on the host machine (outside Docker shell), push dependencies to the Android device:
```
make adb-push
```
You can reuse this target to push the `whisperkit-cli` if you rebuild it. Note that this is not necessary for Linux build.

4. Clean:
```
make clean [all]
```
With `all` option, it will conduct deep clean including open source components.

</details>

# CLI Run and Test

<details>
  <summary> (Click to expand) </summary>

1. Run test on with a sample audio. For Android:
```
make build
```

For Linux:
```
make build linux
```

2. Run on Android with `run_on_android.sh` script:

Log in via adb shell:
```
adb shell
cd /sdcard/argmax/tflite
sh run_on_android.sh
```

3. Manually run `whisperkit-cli`:

Usage: 

```
whisperkit-cli transcribe --model-path /path/to/my/whisper_model --audio-path /path/to/my/audio_file.m4a --report --report-path /path/to/dump/report.json
```

For all options, run `whisperkit-cli --help`

For Android, log in via adb shell:
```
adb shell
cd /sdcard/argmax/tflite
export PATH=/data/local/tmp/bin:$PATH
export LD_LIBRARY_PATH=/data/local/tmp/lib
whisperkit-cli transcribe --model-path  /path/to/openai_whisper-base --audio-path /path/to/inputs/jfk_441khz.m4a
```

4. Sample execution output:
```
root@cf40510e9b93:/src/AXIE# ./build/linux/whisperkit-cli transcribe --model-path /src/AXIE/models/openai_whisper-small --audio-path /src/AXIE/test/jfk_441khz.m4a 
SoC: 	generic CPU (x86, arm64, etc) 
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
postproc vocab size: 51864
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '(null)':
  Metadata:
    major_brand     : M4A 
    minor_version   : 0
    compatible_brands: M4A mp42isom
    creation_time   : 2024-08-07T16:38:45.000000Z
    iTunSMPB        :  00000000 00000840 000000D4 00000000000766EC 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
  Duration: 00:00:11.05, start: 0.047891, bitrate: 73 kb/s
  Stream #0:0[0x1](eng): Audio: aac (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 31 kb/s (default)
      Metadata:
        creation_time   : 2024-08-07T16:38:45.000000Z
        vendor_id       : [0][0][0][0]
Stream: freq - 44100, channels - 1, format - 32784, target_buf size - 1440000
[aac @ 0x55555a5b8c00] Could not update timestamps for skipped samples.
Transcription:   And so, my fellow Americans, ask not what your country can do for you.   Ask what you can do for your country.
```
</details>

# Running Android App

<details>
  <summary> (Click to expand) </summary>

1. Move model to assets folder  
Download the models if you haven't done so as specified in the Installation section. Move specifically the "whisper_tiny" folder into the assets folder of the app. 

2. Run the app with Android Studio

3. Things to consider.  
QNN will only work if the SoC of your device is among the supported ones listed in the C++ code. The model may take a couple minutes to load when using QNN delegate.
After recording with the microphone, the input is saved into the MicInput.wav file, you can select it to transcribe your audio.

</details>

## Contributing

WhisperKit Android is currently in the v0.1 Beta stage.  We are actively developing the project and welcome contributions from the community.

## License
- We release WhisperKit Android under [MIT License](LICENSE).
- SDL3 open-source (audio resampling) is released under [zlib license](https://github.com/libsdl-org/SDL/blob/main/LICENSE.txt)
- FFmpeg open-source (audio decompressing) is released under [LGPL](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)
- OpenAI Whisper model open-source checkpoints were released under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).
- Qualcomm AI Hub `.tflite` models and QNN libraries for NPU deployment are released under the [Qualcomm AI Model & Software License](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

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
