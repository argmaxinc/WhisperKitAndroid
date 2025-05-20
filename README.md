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

- [App](#app)
- [Using WhisperKit in Your Android App](#using-whisperkit-in-your-android-app)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

# Installation

<details>
  <summary> (Click to expand) </summary>

The following setup was tested on macOS 15.1.

1. Ensure you have the required build tools using:

```bash
make setup
```

2. Download Whisper models (<1.5GB) and auxiliary files

```bash
make download-models
```

3. Build development environment in Docker with all development tools (~12GB):

```bash
make env
```

The first time running `make env` command will take several minutes.

After the Docker image builds, the next time running `make env` will execute inside the Docker container right away.

You can use the following to rebuild the Docker image, if needed:

```bash
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

```bash
make env
```

2. Inside the Docker environment, build the `whisperkit-cli` CLI using (for Android and Linux):

```bash
make build [linux | qnn | gpu]
```

The QNN option builds WhisperKit with Qualcomm AI NPU support and the QNN TFLite delegate.
The 'gpu' option is the generic GPU backend for all Android devices from TFLite GPU delegate.
Linux builds are currently CPU-only.

3. Back on the host machine (outside Docker shell), push dependencies to the Android device:

```bash
make adb-push
```

You can reuse this target to push the `whisperkit-cli` if you rebuild it. Note that this is not necessary for Linux build.

4. Clean:

```bash
make clean [all]
```

With `all` option, it will conduct deep clean including open source components.

</details>

# CLI Run and Test

<details>
  <summary> (Click to expand) </summary>

1. Run test on with a sample audio. For Android:

```bash
make build
```

For Linux:

```bash
make build linux
```

2. Manually run `whisperkit-cli`:

Usage:

```bash
whisperkit-cli transcribe --model-path /path/to/my/whisper_model --audio-path /path/to/my/audio_file.m4a --report --report-path /path/to/dump/report.json
```

For all options, run `whisperkit-cli --help`

For Android, log in via adb shell:

```bash
adb shell
cd /sdcard/argmax/tflite
export PATH=/data/local/tmp/bin:$PATH
export LD_LIBRARY_PATH=/data/local/tmp/lib
whisperkit-cli transcribe --model-path  /path/to/openai_whisper-base --audio-path /path/to/inputs/jfk_441khz.m4a
```

3. Sample execution output:

```bash
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

## App

For getting the libraries into the app, follow the instructions above to fetch them.
Then run:

```bash
make build jni
```

To get the models on device, just copy them from WhisperKitAndroid/models and into the WhisperKitAndroid/android/app/src/main/assets directory.

## Using WhisperKit in Your Android App

WhisperKit API is currently experimental and requires explicit opt-in using `@OptIn(ExperimentalWhisperKit::class)`. This indicates that the API may change in future releases. Use with caution in production code.

To use WhisperKit in your Android app, you need to:

1. Add the following dependencies to your app's `build.gradle.kts`:

```kotlin
dependencies {
    // 1. WhisperKit SDK
    implementation("com.argmaxinc:whisperkit:0.3.0")
    
    // 2. QNN dependencies for hardware acceleration
    implementation("com.qualcomm.qnn:qnn-runtime:2.34.0")
    implementation("com.qualcomm.qnn:qnn-litert-delegate:2.34.0")
}
```

2. Configure JNI library packaging in your app's `build.gradle.kts`:

```kotlin
android {
    // ...
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}
```

3. Use WhisperKit in your code:

```kotlin
@OptIn(ExperimentalWhisperKit::class)
class YourActivity : AppCompatActivity() {
    private lateinit var whisperKit: WhisperKit
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize WhisperKit
        // Note: Always use applicationContext to avoid memory leaks and ensure proper lifecycle management
        whisperKit = WhisperKit.Builder()
            .setModel(WhisperKit.OPENAI_TINY_EN)
            .setApplicationContext(applicationContext)
            .setCallback { what, timestamp, msg ->
                // Handle transcription output
                when (what) {
                    WhisperKit.TextOutputCallback.MSG_INIT -> {
                        // Model initialized successfully
                    }
                    WhisperKit.TextOutputCallback.MSG_TEXT_OUT -> {
                        // New transcription available
                        val text = msg
                        val time = timestamp
                        // Process the transcribed text as it becomes available
                        // This callback will be called multiple times as more audio is processed
                    }
                    WhisperKit.TextOutputCallback.MSG_CLOSE -> {
                        // Cleanup complete
                    }
                }
            }
            .build()
            
        // Load the model
        lifecycleScope.launch {
            whisperKit.loadModel().collect { progress ->
                // Handle download progress
            }
            
            // Initialize with audio parameters
            whisperKit.init(frequency = 16000, channels = 1, duration = 0)
            
            // Transcribe audio data in chunks
            // You can call transcribe() multiple times with different chunks of audio data
            // Results will be delivered through the callback as they become available
            val audioChunk1: ByteArray = // First chunk of audio data
            whisperKit.transcribe(audioChunk1)
            
            val audioChunk2: ByteArray = // Second chunk of audio data
            whisperKit.transcribe(audioChunk2)
            
            // Continue processing more chunks as needed...
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        whisperKit.deinitialize()
    }
}
```

Note: The WhisperKit API is currently experimental and may change in future releases. Make sure to handle the `@OptIn(ExperimentalWhisperKit::class)` annotation appropriately in your code.

## Contributing

WhisperKit Android is currently in the beta stage. We are actively developing the project and welcome contributions from the community.

## License

- We release WhisperKit Android under [MIT License](LICENSE).
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
   URL = {https://github.com/argmaxinc/WhisperKitAndroid}
}
```
