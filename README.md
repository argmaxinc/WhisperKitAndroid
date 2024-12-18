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

ArgmaX Inference Engine (`axie`) orchestration for TFLite is provided as the `axie_tflite` CLI.

1. Execute into the Docker build environment:
```
make env
```

2. Inside the Docker environment, build the `axie_tflite` CLI using:
```
make build
```

3. On the host machine (outside Docker shell), push dependencies to the Android device:
```
make adb-push
```
You can reuse this target to push the `axie_tflite` if you rebuild it.

If you want to include audio files, place them in the `/path/to/WhisperKitAndroid/inputs` folder and they will be copied to `/sdcard/argmax/tflite/inputs/`.

4. Connect to the Android device using:
```
make adb-shell
```

5. Run `axie_tflite`
```
Usage: axie_tflite <audio input> <tiny | base | small>
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