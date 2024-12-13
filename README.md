<div align="center">

<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# WhisperKit Android (Alpha)
</div>

WhisperKit Android brings Foundation Models On Device for Automatic Speech Recognition. It extends the performance and feature set of [WhisperKit](https://github.com/argmaxinc/WhisperKit) from Apple platforms to Android and (soon) Linux.

[Example App (Coming with Beta)] [[Blog Post]](https://takeargmax.com/blog/android) [[Python Tools Repo]](https://github.com/argmaxinc/whisperkittools)


## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [CLI Run and Test](#cli-run-and-test)
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

After Docker image builds, the next time running make env will execute inside the Docker container right away.

You can use the following to rebuild the Docker image, if needed:
```
make rebuild-env
```

</details>

# Getting Started

<details>
  <summary> (Click to expand) </summary>

ArgmaX Inference Engine (`AXIE`) orchestration for TFLite is provided as the `whisperax_cli` CLI.

1. Execute into the Docker build environment:
```
make env
```

2. Inside the Docker environment, build the `whisperax_cli` CLI using (for Android and Linux):
```
make build [linux]
```

3. Back on the host machine (outside Docker shell), push dependencies to the Android device:
```
make adb-push
```
You can reuse this target to push the `whisperax_cli` if you rebuild it. Note that this is not necessary for Linux build.

If you want to include audio files, place them in the `/path/to/WhisperKitAndroid/inputs` folder and they will be copied to `/sdcard/argmax/tflite/inputs/`.

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
make test
```
For linux:
```
make env
make test linux
```

2. Manually run `whisperax_cli`:
```
Usage: whisperax_cli <audio input> <tiny | base | small>
```
For Android, log in via adb shell:
```
adb shell
cd /sdcard/argmax/tflite
export PATH=/data/local/tmp/bin:$PATH
export LD_LIBRARY_PATH=/data/local/tmp/lib
whisperax_cli ./inputs/jfk_441khz.m4a base
```

3. Sample execution output:
```
Argmax, Inc.

Audio Codec: aac
tflite_init input: {"cache":"/data/local/tmp/cache","ch":1,"debug":false,"dur":11052,"fmt":1,"freq":44100,"lib":"/data/local/tmp/lib","root_path":"/storage/emulated/0/argmax/tflite","size":"tiny"}
SoC: 	QCM5430 -> TFLite GPU
root dir:	/storage/emulated/0/argmax/tflite
lib dir:	/data/local/tmp/lib
cache dir:	/data/local/tmp/cache
Stream: freq - 44100, channels - 1, format - 32784, target_buf size - 1440000
postproc vocab size: 51864
tflite_init done..

Final Text:   And so my fellow American asked not what your country can do for you   ask what you can do for your country.
Deleted interpreter & delegate for post_proc
Deleted interpreter & delegate for whisper_decoder
Deleted interpreter & delegate for whisper_encoder
Deleted interpreter & delegate for mel_spectrogram
Deleted interpreter & delegate for audio_input
tflite_close done..

Model latencies:
  Audio Input: 1 inferences,	 median:2.12 ms
  Melspectro: 1 inferences,	 median:74.50 ms
  Encoder: 1 inferences,	 median:1197.14 ms
  Decoder: 28 inferences,	 median:41.62 ms
  Postproc: 28 inferences,	 median:1.23 ms
=========================
Total Duration:	 3262.000 ms
```
</details>

## Contributing & Roadmap

WhisperKit Android is currently in the v0.1 Alpha stage. Contributions from the community will be encouraged after the project reaches the v0.1 Beta milestone.

### v0.1 Beta (November 2024)
- [ ] Temperature fallbacks for decoding guardrails
- [ ] Input audio file format coverage for wav, flac, mp4, m4a, mp3
- [ ] Output file format coverage for SRT, VTT, and OpenAI-compatible JSON
- [ ] [WhisperKit Benchmarks](https://huggingface.co/spaces/argmaxinc/whisperkit-evals) performance and quality data publication

### v0.2 (Q1 2025)
- [ ] Whisper Large v3 Turbo (v20240930) support
- [ ] Streaming real-time inference
- [ ] Model compression

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