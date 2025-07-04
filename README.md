<div align="center">

<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# WhisperKit Android

[![Tests](https://github.com/argmaxinc/whisperkitandroid/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/argmaxinc/whisperkitandroid/actions/workflows/pr-checks.yml)
[![License](https://img.shields.io/github/license/argmaxinc/whisperkitandroid?logo=github&logoColor=969da4&label=License&labelColor=353a41&color=32d058)](LICENSE.md)
[![Maven Central](https://img.shields.io/maven-central/v/com.argmaxinc/whisperkit?logo=sonatype&logoColor=969da4&label=Maven%20Central&labelColor=353a41&color=32d058)](https://central.sonatype.com/artifact/com.argmaxinc/whisperkit)
[![Discord](https://img.shields.io/discord/1171912382512115722?style=flat&logo=discord&logoColor=969da4&label=Discord&labelColor=353a41&color=32d058&link=https%3A%2F%2Fdiscord.gg%2FG5F5GZGecC)](https://discord.gg/G5F5GZGecC)

</div>

WhisperKit Android brings Foundation Models On Device for Automatic Speech Recognition. It extends the performance and feature set of [WhisperKit](https://github.com/argmaxinc/WhisperKit) from Apple platforms to Android and Linux.  The current feature set is a subset of the iOS counterpart,
but we are continuing to invest in Android and now welcome contributions from the community.

[[Example App]](https://play.google.com/store/apps/details?id=com.argmaxinc.whisperax) [[Blog Post]](https://takeargmax.com/blog/android) [[Python Tools Repo]](https://github.com/argmaxinc/whisperkittools)

## Table of Contents

- [Getting Started](#getting-started)
- [Build From Source](#build-from-source)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

# Getting Started

WhisperKit API is currently experimental and requires explicit opt-in using `@OptIn(ExperimentalWhisperKit::class)`. This indicates that the API may change in future releases. Use with caution in production code.

To use WhisperKit in your Android app, you need to:

1. Add the following dependencies to your app's `build.gradle.kts`:

```kotlin
dependencies {
   // 1. WhisperKit SDK
   implementation("com.argmaxinc:whisperkit:0.3.2")  // Check badge above for latest version

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
         .setCallback { what, result ->
            // Handle transcription output
            when (what) {
               WhisperKit.TextOutputCallback.MSG_INIT -> {
                  // Model initialized successfully
               }
               WhisperKit.TextOutputCallback.MSG_TEXT_OUT -> {
                  // New transcription available
                  val fullText = result.text
                  val segments = result.segments
                  // Process the transcribed text as it becomes available
                  // This callback will be called multiple times as more audio is processed
                  segments.forEach { segment ->
                     // Process each segment
                     val segmentText = segment.text
                  }
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

# Build From Source

<details>
  <summary> (Click to expand) </summary>

The following setup was tested on macOS 15.1.

## Common Setup Steps

These steps are required for both Android app development and CLI:

1. Install required build tools:

```bash
make setup
```

2. Build development environment in Docker with all development tools:

```bash
make env
```

The first time running `make env` command will take several minutes. After the Docker image builds, the next time running `make env` will execute inside the Docker container right away.

If you need to rebuild the Docker image:
```bash
make rebuild-env
```

### Android App Development Path

1. Build and enter the Docker environment:

```bash
make env
```

2. Build the required native libraries:

```bash
make build jni
```

3. Open the Android project in Android Studio:
   - Open the root project in Android Studio
   - Navigate to `android/examples/WhisperAX`
   - Build and run the app

### CLI Development Path

1. Build and enter the Docker environment:

```bash
make env
```

2. Build the CLI app:

```bash
make build [linux | qnn | gpu]
```

- `linux`: CPU-only build for Linux
- `qnn`: Android build with Qualcomm NPU support
- `gpu`: Android build with GPU support

3. Push dependencies to Android device (skip for Linux):

```bash
make adb-push
```

4. Run the CLI app:

For Android:

```bash
adb shell
cd /sdcard/argmax/tflite
export PATH=/data/local/tmp/bin:$PATH
export LD_LIBRARY_PATH=/data/local/tmp/lib
whisperkit-cli transcribe --model-path /path/to/openai_whisper-base --audio-path /path/to/inputs/jfk_441khz.m4a
```

For Linux:

```bash
./build/linux/whisperkit-cli transcribe --model-path /path/to/my/whisper_model --audio-path /path/to/my/audio_file.m4a --report --report-path /path/to/dump/report.json
```

For all options, run `whisperkit-cli --help`

5. Clean build files when needed:

```bash
make clean [all]
```
With `all` option, it will conduct deep clean including open source components.

</details>

# Contributing

We are actively developing the project and welcome contributions from the community.

## Development Setup

### Installing clang-format

The project uses clang-format 14 for C++ code formatting. You'll need to install it based on your operating system:

#### macOS
```bash
# Install LLVM 14
brew install llvm@14

# Create symlink for clang-format
sudo ln -sf /opt/homebrew/opt/llvm@14/bin/clang-format /opt/homebrew/bin/clang-format
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y clang-format-14
sudo ln -sf /usr/bin/clang-format-14 /usr/local/bin/clang-format
```

Verify the installation:
```bash
clang-format --version
```

To check C++ code formatting:
```bash
./gradlew spotlessCppCheck
```

## Git Hooks

This project uses Git hooks to maintain code quality. These hooks help ensure consistent code formatting and quality standards.

### Setup

To use the Git hooks, run the following command in your repository root:

```bash
git config core.hooksPath .githooks
```

### Available Hooks

#### pre-commit
- Runs `spotlessApply` to automatically fix code formatting issues
- If formatting fixes are applied, they are automatically staged
- The commit will be blocked if `spotlessApply` fails

### Troubleshooting

If you need to bypass the hooks temporarily (not recommended), you can use:
```bash
git commit --no-verify
```

# License

- We release WhisperKit Android under [MIT License](LICENSE).
- FFmpeg open-source (audio decompressing) is released under [LGPL](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)
- OpenAI Whisper model open-source checkpoints were released under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).
- Qualcomm AI Hub `.tflite` models and QNN libraries for NPU deployment are released under the [Qualcomm AI Model & Software License](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

# Citation

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
