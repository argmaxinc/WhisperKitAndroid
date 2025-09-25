@file:OptIn(ExperimentalWhisperKit::class)

package com.argmaxinc.whisperkit.network

import com.argmaxinc.whisperkit.ExperimentalWhisperKit
import com.argmaxinc.whisperkit.WhisperKit
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi
import com.argmaxinc.whisperkit.huggingface.Repo
import com.argmaxinc.whisperkit.huggingface.RepoType
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ArgmaxModelDownloaderImplTest {
    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var root: File
    private lateinit var huggingFaceApi: HuggingFaceApi
    private lateinit var downloader: ArgmaxModelDownloaderImpl

    @Before
    fun setup() {
        root = tempFolder.newFolder()
        huggingFaceApi = mockk()
        downloader = ArgmaxModelDownloaderImpl(huggingFaceApi)
    }

    private fun testDownload(
        modelVariant: String,
        expectedTokenizerRepo: String,
        expectedEncoderDecoderRepo: String,
        expectedEncoderDecoderGlobFilters: List<String>,
        expectedMelSpectrogramPath: String,
        cached: Boolean = false,
    ) {
        val qualcommModel = modelVariant.startsWith("qualcomm/")

        // Mock config metadata
        coEvery {
            huggingFaceApi.getFileMetadata(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                filename = eq("config.json"),
            )
        } returns HuggingFaceApi.FileMetadata(500L, "config.json")

        // Mock tokenizer metadata
        coEvery {
            huggingFaceApi.getFileMetadata(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                filename = eq("tokenizer.json"),
            )
        } returns HuggingFaceApi.FileMetadata(1000L, "tokenizer.json")
        // Mock encoder/decoder metadata
        coEvery {
            huggingFaceApi.getFileMetadata(
                from =
                eq(
                    if (qualcommModel) {
                        Repo(
                            expectedEncoderDecoderRepo,
                            RepoType.MODELS,
                        )
                    } else {
                        Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)
                    },
                ),
                revision = any(),
                globFilters = eq(expectedEncoderDecoderGlobFilters),
            )
        } returns
            listOf(
                HuggingFaceApi.FileMetadata(
                    2000L,
                    if (qualcommModel) "WhisperEncoder.tflite" else "$expectedMelSpectrogramPath/AudioEncoder.tflite",
                ),
                HuggingFaceApi.FileMetadata(
                    3000L,
                    if (qualcommModel) "WhisperDecoder.tflite" else "$expectedMelSpectrogramPath/TextDecoder.tflite",
                ),
            )
        // Mock mel spectrogram metadata
        coEvery {
            huggingFaceApi.getFileMetadata(
                from = eq(Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)),
                revision = eq("main"),
                filename = eq("$expectedMelSpectrogramPath/MelSpectrogram.tflite"),
            )
        } returns
            HuggingFaceApi.FileMetadata(
                4000L,
                "$expectedMelSpectrogramPath/MelSpectrogram.tflite",
            )

        if (cached) {
            testCached(
                modelVariant,
                qualcommModel,
                expectedMelSpectrogramPath,
            )
        } else {
            testNonCached(
                modelVariant,
                qualcommModel,
                expectedMelSpectrogramPath,
                expectedTokenizerRepo,
                expectedEncoderDecoderRepo,
                expectedEncoderDecoderGlobFilters,
            )
        }
    }

    private fun testCached(
        modelVariant: String,
        qualcommModel: Boolean,
        expectedMelSpectrogramPath: String,
    ) {
        // Create cached files upfront
        // 1. Create config.json
        val configFile = File(root, "config.json")
        configFile.createNewFile()
        configFile.setWritable(true)
        configFile.setReadable(true)
        configFile.setExecutable(true)
        configFile.setLastModified(System.currentTimeMillis())
        configFile.writeBytes(ByteArray(500))

        // 2. Create tokenizer.json
        val tokenizerFile = File(root, "tokenizer.json")
        tokenizerFile.createNewFile()
        tokenizerFile.setWritable(true)
        tokenizerFile.setReadable(true)
        tokenizerFile.setExecutable(true)
        tokenizerFile.setLastModified(System.currentTimeMillis())
        tokenizerFile.writeBytes(ByteArray(1000)) // Set size to match metadata

        // 3. Create encoder/decoder files
        val audioEncoderFile = File(root, "AudioEncoder.tflite")
        val textDecoderFile = File(root, "TextDecoder.tflite")
        audioEncoderFile.createNewFile()
        textDecoderFile.createNewFile()
        audioEncoderFile.writeBytes(ByteArray(2000)) // Set size to match metadata
        textDecoderFile.writeBytes(ByteArray(3000)) // Set size to match metadata

        // 4. Create MelSpectrogram.tflite
        val melSpectrogramFile = File(root, "MelSpectrogram.tflite")
        melSpectrogramFile.createNewFile()
        melSpectrogramFile.writeBytes(ByteArray(4000)) // Set size to match metadata

        runBlocking {
            downloader.download(ArgmaxModel.WHISPER, modelVariant, root).collect {}
        }

        // Verify snapshot was never called when files are cached
        verify(exactly = 0) {
            huggingFaceApi.snapshot(
                from = any(),
                revision = any(),
                globFilters = any(),
                baseDir = any(),
            )
        }

        if (qualcommModel) {
            // Verify WhisperEncoder.tflite and WhisperDecoder.tflite
            // renamed to AudioEncoder.tflite and TextDecoder.tflite
            assert(
                !File(
                    root,
                    "WhisperEncoder.tflite",
                ).exists(),
            ) { "Original WhisperEncoder.tflite should not exist" }
            assert(
                !File(
                    root,
                    "WhisperDecoder.tflite",
                ).exists(),
            ) { "Original WhisperDecoder.tflite should not exist" }
            assert(
                File(
                    root,
                    "AudioEncoder.tflite",
                ).exists(),
            ) { "AudioEncoder.tflite should exist" }
            assert(File(root, "TextDecoder.tflite").exists()) { "TextDecoder.tflite should exist" }
        } else {
            // Verify expectedEncoderDecoderRepo/AudioEncoder.tflite is moved to AudioEncoder.tflite
            //    expectedEncoderDecoderRepo/TextDecoder.tflite is moved to TextDecoder.tflite
            assert(
                !File(
                    root,
                    expectedMelSpectrogramPath,
                ).exists(),
            ) { "Original model directory should be deleted" }
            assert(
                File(
                    root,
                    "AudioEncoder.tflite",
                ).exists(),
            ) { "AudioEncoder.tflite should exist" }
            assert(File(root, "TextDecoder.tflite").exists()) { "TextDecoder.tflite should exist" }
        }

        // after download, the file should be moved to root/MelSpectrogram.tflite
        assert(
            File(
                root,
                "MelSpectrogram.tflite",
            ).exists(),
        ) { "MelSpectrogram.tflite should exist in root directory" }

        // Add directory deletion verification
        if (!qualcommModel) {
            assert(!File(root, expectedMelSpectrogramPath).exists()) {
                "Model directory should be deleted after download"
            }
        }
    }

    private fun testNonCached(
        modelVariant: String,
        qualcommModel: Boolean,
        expectedMelSpectrogramPath: String,
        expectedTokenizerRepo: String,
        expectedEncoderDecoderRepo: String,
        expectedEncoderDecoderGlobFilters: List<String>,
    ) {
        // Mock config snapshot
        every {
            huggingFaceApi.snapshot(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("config.json")),
                baseDir = eq(root),
            )
        } returns flowOf(HuggingFaceApi.Progress(1.0f))

        // Mock tokenizer snapshot
        every {
            huggingFaceApi.snapshot(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("tokenizer.json")),
                baseDir = eq(root),
            )
        } returns flowOf(HuggingFaceApi.Progress(1.0f))

        // Mock encoder/decoder snapshot
        every {
            huggingFaceApi.snapshot(
                from = eq(Repo(expectedEncoderDecoderRepo, RepoType.MODELS)),
                revision = any(),
                globFilters = eq(expectedEncoderDecoderGlobFilters),
                baseDir = eq(root),
            )
        } returns flowOf(HuggingFaceApi.Progress(1.0f))

        // Mock mel spectrogram snapshot
        every {
            huggingFaceApi.snapshot(
                from = eq(Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("$expectedMelSpectrogramPath/MelSpectrogram.tflite")),
                baseDir = eq(root),
            )
        } returns flowOf(HuggingFaceApi.Progress(1.0f))

        // Mock download success - write the original files
        if (qualcommModel) {
            // for Qualcomm Models, encoder/decoder downloaded as WhisperEncoder.tflite and WhisperDecoder.tflite
            //  Create the original files, mocking download success
            val whisperEncoderFile = File(root, "WhisperEncoder.tflite")
            val whisperDecoderFile = File(root, "WhisperDecoder.tflite")
            whisperEncoderFile.createNewFile()
            whisperDecoderFile.createNewFile()
        } else {
            // for openai model, need to move $modelDir/AudioEncoder.tflite
            // and $modelDir/TextDecoder.tflite to AudioEncoder.tflite TextDecoder.tflite
            //  modelDir is the same as expectedMelSpectrogramPath
            //  Create the original files, mocking download success
            val modelDir = File(root, expectedMelSpectrogramPath)
            modelDir.mkdirs()
            File(modelDir, "AudioEncoder.tflite").createNewFile()
            File(modelDir, "TextDecoder.tflite").createNewFile()
        }

        // Mock the original MelSpectrogram.tflite download location
        // root/expectedMelSpectrogramRepo/MelSpectrogram.tflite
        val originalMelSpectrogramDir = File(root, expectedMelSpectrogramPath)
        originalMelSpectrogramDir.mkdirs()
        val originalMelSpectrogramFile = File(originalMelSpectrogramDir, "MelSpectrogram.tflite")
        originalMelSpectrogramFile.createNewFile()

        runBlocking {
            downloader.download(ArgmaxModel.WHISPER, modelVariant, root).collect {}
        }

        // Verify snapshot was called exactly 4 times for each component
        verify(exactly = 1) {
            huggingFaceApi.snapshot(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("config.json")),
                baseDir = eq(root),
            )
        }
        verify(exactly = 1) {
            huggingFaceApi.snapshot(
                from = eq(Repo(expectedTokenizerRepo, RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("tokenizer.json")),
                baseDir = eq(root),
            )
        }
        verify(exactly = 1) {
            huggingFaceApi.snapshot(
                from =
                eq(
                    if (qualcommModel) {
                        Repo(
                            expectedEncoderDecoderRepo,
                            RepoType.MODELS,
                        )
                    } else {
                        Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)
                    },
                ),
                revision = any(),
                globFilters = eq(expectedEncoderDecoderGlobFilters),
                baseDir = eq(root),
            )
        }
        verify(exactly = 1) {
            huggingFaceApi.snapshot(
                from = eq(Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)),
                revision = eq("main"),
                globFilters = eq(listOf("$expectedMelSpectrogramPath/MelSpectrogram.tflite")),
                baseDir = eq(root),
            )
        }
        if (qualcommModel) {
            // Verify WhisperEncoder.tflite and WhisperDecoder.tflite renamed
            // to AudioEncoder.tflite and TextDecoder.tflite
            assert(
                !File(
                    root,
                    "WhisperEncoder.tflite",
                ).exists(),
            ) { "Original WhisperEncoder.tflite should not exist" }
            assert(
                !File(
                    root,
                    "WhisperDecoder.tflite",
                ).exists(),
            ) { "Original WhisperDecoder.tflite should not exist" }
            assert(
                File(
                    root,
                    "AudioEncoder.tflite",
                ).exists(),
            ) { "AudioEncoder.tflite should exist" }
            assert(File(root, "TextDecoder.tflite").exists()) { "TextDecoder.tflite should exist" }
        } else {
            // Verify expectedEncoderDecoderRepo/AudioEncoder.tflite is moved to AudioEncoder.tflite
            //    expectedEncoderDecoderRepo/TextDecoder.tflite is moved to TextDecoder.tflite
            assert(
                !File(
                    root,
                    expectedMelSpectrogramPath,
                ).exists(),
            ) { "Original model directory should be deleted" }
            assert(
                File(
                    root,
                    "AudioEncoder.tflite",
                ).exists(),
            ) { "AudioEncoder.tflite should exist" }
            assert(File(root, "TextDecoder.tflite").exists()) { "TextDecoder.tflite should exist" }
        }

        // Verify MelSpectrogram.tflite was moved and directory was cleaned up
        assert(!originalMelSpectrogramFile.exists()) { "Original MelSpectrogram.tflite should not exist" }
        assert(!originalMelSpectrogramDir.exists()) { "Original model directory should be deleted" }
        // after download, the file should be moved to root/MelSpectrogram.tflite
        assert(
            File(
                root,
                "MelSpectrogram.tflite",
            ).exists(),
        ) { "MelSpectrogram.tflite should exist in root directory" }

        // Add directory deletion verification
        if (!qualcommModel) {
            assert(!File(root, expectedMelSpectrogramPath).exists()) {
                "Model directory should be deleted after download"
            }
        }
    }

    @Test
    fun `download creates correct flows for OpenAI tiny model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_TINY_EN,
            expectedTokenizerRepo = "openai/whisper-tiny.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-tiny.en/AudioEncoder.tflite",
                "openai_whisper-tiny.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-tiny.en",
        )

    @Test
    fun `download creates correct flows for OpenAI tiny multilingual model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_TINY,
            expectedTokenizerRepo = "openai/whisper-tiny",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-tiny/AudioEncoder.tflite",
                "openai_whisper-tiny/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-tiny",
        )

    @Test
    fun `download creates correct flows for OpenAI base model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_BASE_EN,
            expectedTokenizerRepo = "openai/whisper-base.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-base.en/AudioEncoder.tflite",
                "openai_whisper-base.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-base.en",
        )

    @Test
    fun `download creates correct flows for OpenAI base multilingual model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_BASE,
            expectedTokenizerRepo = "openai/whisper-base",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-base/AudioEncoder.tflite",
                "openai_whisper-base/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-base",
        )

    @Test
    fun `download creates correct flows for OpenAI small model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_SMALL_EN,
            expectedTokenizerRepo = "openai/whisper-small.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-small.en/AudioEncoder.tflite",
                "openai_whisper-small.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-small.en",
        )

    @Test
    fun `download creates correct flows for Qualcomm tiny model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_TINY_EN,
            expectedTokenizerRepo = "openai/whisper-tiny.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Tiny-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-tiny.en",
        )

    @Test
    fun `download creates correct flows for Qualcomm base model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_BASE_EN,
            expectedTokenizerRepo = "openai/whisper-base.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Base-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-base.en",
        )

    @Test
    fun `download creates correct flows for Qualcomm small model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_SMALL_EN,
            expectedTokenizerRepo = "openai/whisper-small.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Small-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-small.en",
        )

    @Test
    fun `download uses cached files when available for OpenAI tiny model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_TINY_EN,
            expectedTokenizerRepo = "openai/whisper-tiny.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-tiny.en/AudioEncoder.tflite",
                "openai_whisper-tiny.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-tiny.en",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for OpenAI tiny multilingual model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_TINY,
            expectedTokenizerRepo = "openai/whisper-tiny",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-tiny/AudioEncoder.tflite",
                "openai_whisper-tiny/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-tiny",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for OpenAI base model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_BASE_EN,
            expectedTokenizerRepo = "openai/whisper-base.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-base.en/AudioEncoder.tflite",
                "openai_whisper-base.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-base.en",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for OpenAI base multilingual model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_BASE,
            expectedTokenizerRepo = "openai/whisper-base",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-base/AudioEncoder.tflite",
                "openai_whisper-base/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-base",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for OpenAI small model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.OPENAI_SMALL_EN,
            expectedTokenizerRepo = "openai/whisper-small.en",
            expectedEncoderDecoderRepo = "argmaxinc/whisperkit-litert",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "openai_whisper-small.en/AudioEncoder.tflite",
                "openai_whisper-small.en/TextDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "openai_whisper-small.en",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for Qualcomm tiny model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_TINY_EN,
            expectedTokenizerRepo = "openai/whisper-tiny.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Tiny-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-tiny.en",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for Qualcomm base model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_BASE_EN,
            expectedTokenizerRepo = "openai/whisper-base.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Base-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-base.en",
            cached = true,
        )

    @Test
    fun `download uses cached files when available for Qualcomm small model`() =
        testDownload(
            modelVariant = WhisperKit.Builder.QUALCOMM_SMALL_EN,
            expectedTokenizerRepo = "openai/whisper-small.en",
            expectedEncoderDecoderRepo = "qualcomm/Whisper-Small-En",
            expectedEncoderDecoderGlobFilters =
            listOf(
                "WhisperEncoder.tflite",
                "WhisperDecoder.tflite",
            ),
            expectedMelSpectrogramPath = "quic_openai_whisper-small.en",
            cached = true,
        )

    @Test(expected = IllegalArgumentException::class)
    fun `download throws IllegalArgumentException for invalid variant`() =
        runTest {
            downloader.download(ArgmaxModel.WHISPER, "invalid-variant", root)
        }

    @Test
    fun `download combines progress from all flows correctly`() =
        runTest {
            val configProgress = 0.2f
            val tokenizerProgress = 0.3f
            val encoderDecoderProgress = 0.6f
            val melSpectrogramProgress = 0.9f
            val expectedProgress =
                (configProgress + tokenizerProgress + encoderDecoderProgress + melSpectrogramProgress) / 4.0f

            // Mock config metadata
            coEvery {
                huggingFaceApi.getFileMetadata(
                    from = eq(Repo("openai/whisper-tiny.en", RepoType.MODELS)),
                    revision = eq("main"),
                    filename = eq("config.json"),
                )
            } returns HuggingFaceApi.FileMetadata(500L, "config.json")

            // Mock tokenizer metadata
            coEvery {
                huggingFaceApi.getFileMetadata(
                    from = eq(Repo("openai/whisper-tiny.en", RepoType.MODELS)),
                    revision = eq("main"),
                    filename = eq("tokenizer.json"),
                )
            } returns HuggingFaceApi.FileMetadata(1000L, "tokenizer.json")

            // Mock encoder/decoder metadata
            coEvery {
                huggingFaceApi.getFileMetadata(
                    from = eq(Repo("qualcomm/Whisper-Tiny-En", RepoType.MODELS)),
                    revision = any(),
                    globFilters = eq(listOf("WhisperEncoder.tflite", "WhisperDecoder.tflite")),
                )
            } returns
                listOf(
                    HuggingFaceApi.FileMetadata(2000L, "WhisperEncoder.tflite"),
                    HuggingFaceApi.FileMetadata(3000L, "WhisperDecoder.tflite"),
                )

            // Mock mel spectrogram metadata
            coEvery {
                huggingFaceApi.getFileMetadata(
                    from = eq(Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)),
                    revision = eq("main"),
                    filename = eq("quic_openai_whisper-tiny.en/MelSpectrogram.tflite"),
                )
            } returns
                HuggingFaceApi.FileMetadata(
                    4000L,
                    "quic_openai_whisper-tiny.en/MelSpectrogram.tflite",
                )

            // Mock tokenizer snapshot
            every {
                huggingFaceApi.snapshot(
                    from = eq(Repo("openai/whisper-tiny.en", RepoType.MODELS)),
                    revision = eq("main"),
                    globFilters = eq(listOf("config.json")),
                    baseDir = eq(root),
                )
            } returns flowOf(HuggingFaceApi.Progress(configProgress))

            // Mock encoder/decoder snapshot
            every {
                huggingFaceApi.snapshot(
                    from = eq(Repo("openai/whisper-tiny.en", RepoType.MODELS)),
                    revision = eq("main"),
                    globFilters = eq(listOf("tokenizer.json")),
                    baseDir = eq(root),
                )
            } returns flowOf(HuggingFaceApi.Progress(tokenizerProgress))

            // Mock encoder/decoder snapshot
            every {
                huggingFaceApi.snapshot(
                    from = eq(Repo("qualcomm/Whisper-Tiny-En", RepoType.MODELS)),
                    revision = any(),
                    globFilters = eq(listOf("WhisperEncoder.tflite", "WhisperDecoder.tflite")),
                    baseDir = eq(root),
                )
            } returns flowOf(HuggingFaceApi.Progress(encoderDecoderProgress))

            // Mock mel spectrogram snapshot
            every {
                huggingFaceApi.snapshot(
                    from = eq(Repo("argmaxinc/whisperkit-litert", RepoType.MODELS)),
                    revision = eq("main"),
                    globFilters = eq(listOf("quic_openai_whisper-tiny.en/MelSpectrogram.tflite")),
                    baseDir = eq(root),
                )
            } returns flowOf(HuggingFaceApi.Progress(melSpectrogramProgress))

            // Collect the progress values
            val progressValues = mutableListOf<Float>()
            runBlocking {
                downloader.download(
                    ArgmaxModel.WHISPER,
                    WhisperKit.Builder.QUALCOMM_TINY_EN,
                    root,
                )
                    .collect { progress ->
                        progressValues.add(progress.fractionCompleted)
                    }
            }

            // Verify the final progress is the average of all three flows
            assert(progressValues.isNotEmpty()) { "Should have received at least one progress update" }
            assert(progressValues.last() == expectedProgress) {
                "Expected progress $expectedProgress, but got ${progressValues.last()}"
            }
        }
}
