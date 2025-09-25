//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.network

import android.util.Log
import com.argmaxinc.whisperkit.ExperimentalWhisperKit
import com.argmaxinc.whisperkit.WhisperKit
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApiConfig
import com.argmaxinc.whisperkit.huggingface.HuggingFaceLogger
import com.argmaxinc.whisperkit.huggingface.KtorHuggingFaceApiImpl
import com.argmaxinc.whisperkit.huggingface.Repo
import com.argmaxinc.whisperkit.huggingface.RepoType
import com.argmaxinc.whisperkit.network.ArgmaxModel.WHISPER
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.onCompletion
import java.io.File

/**
 * Enum representing the available on-device models provided by Argmax.
 * Currently supports:
 * - [WHISPER]: The Whisper speech recognition model
 */
enum class ArgmaxModel {
    WHISPER,
}

/**
 * Android-specific implementation of [HuggingFaceLogger] that uses Android's Log system.
 * @property tag The tag to use for all log messages
 */
class AndroidHuggingFaceLogger(private val tag: String) : HuggingFaceLogger {
    override fun info(message: String) {
        Log.i(tag, message)
    }

    override fun error(message: String) {
        Log.e(tag, message)
    }

    override fun error(
        throwable: Throwable,
        message: String,
    ) {
        Log.e(tag, message, throwable)
    }
}

/**
 * A utility class for downloading Argmax models from HuggingFace.
 *
 * This class provides functionality to download required model files for a specific variant using
 * HuggingFaceApi, supporting automatic retries and progress reporting during downloads.
 */
internal class ArgmaxModelDownloaderImpl(
    private val huggingFaceApi: HuggingFaceApi =
        KtorHuggingFaceApiImpl(
            config =
            HuggingFaceApiConfig(
                logger = AndroidHuggingFaceLogger("ArgmaxModelDownloader"),
            ),
        ),
) : ArgmaxModelDownloader {
    companion object {
        private const val TOKENIZER_REPO = "TOKENIZER_REPO"
        private const val CONFIG_REPO = "TOKENIZER_REPO"
        private const val ENCODER_DECODER_REPO = "ENCODER_DECODER_REPO"
        private const val ENCODER_DECODER_REVISION = "ENCODER_DECODER_REVISION"

        // dir path under argmaxinc/whisperkit-litert to look up for MelSpectrogram.tflite
        private const val FEATURE_EXTRACTOR_PATH = "FEATURE_EXTRACTOR_PATH"

        @OptIn(ExperimentalWhisperKit::class)
        private val modelConfigs =
            mapOf(
                WhisperKit.Builder.OPENAI_TINY_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-tiny.en",
                        TOKENIZER_REPO to "openai/whisper-tiny.en",
                        ENCODER_DECODER_REPO to "openai_whisper-tiny.en",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "openai_whisper-tiny.en"
                    ),
                WhisperKit.Builder.OPENAI_BASE_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-base.en",
                        TOKENIZER_REPO to "openai/whisper-base.en",
                        ENCODER_DECODER_REPO to "openai_whisper-base.en",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "openai_whisper-base.en"
                    ),
                WhisperKit.Builder.OPENAI_TINY to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-tiny",
                        TOKENIZER_REPO to "openai/whisper-tiny",
                        ENCODER_DECODER_REPO to "openai_whisper-tiny",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "openai_whisper-tiny"
                    ),
                WhisperKit.Builder.OPENAI_BASE to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-base",
                        TOKENIZER_REPO to "openai/whisper-base",
                        ENCODER_DECODER_REPO to "openai_whisper-base",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "openai_whisper-base"
                    ),
                WhisperKit.Builder.OPENAI_SMALL_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-small.en",
                        TOKENIZER_REPO to "openai/whisper-small.en",
                        ENCODER_DECODER_REPO to "openai_whisper-small.en",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "openai_whisper-small.en"
                    ),
                WhisperKit.Builder.QUALCOMM_TINY_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-tiny.en",
                        TOKENIZER_REPO to "openai/whisper-tiny.en",
                        ENCODER_DECODER_REPO to "qualcomm/Whisper-Tiny-En",
                        ENCODER_DECODER_REVISION to "8309cf4d4c30c69132f4f5e83ca8dcb7c17407ae",
                        FEATURE_EXTRACTOR_PATH to "quic_openai_whisper-tiny.en"
                    ),
                WhisperKit.Builder.QUALCOMM_BASE_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-base.en",
                        TOKENIZER_REPO to "openai/whisper-base.en",
                        ENCODER_DECODER_REPO to "qualcomm/Whisper-Base-En",
                        ENCODER_DECODER_REVISION to "4bc89f2f841ee034383a543b954a432febf10ccc",
                        FEATURE_EXTRACTOR_PATH to "quic_openai_whisper-base.en"
                    ),
                WhisperKit.Builder.QUALCOMM_SMALL_EN to
                    mapOf(
                        CONFIG_REPO to "openai/whisper-small.en",
                        TOKENIZER_REPO to "openai/whisper-small.en",
                        ENCODER_DECODER_REPO to "qualcomm/Whisper-Small-En",
                        ENCODER_DECODER_REVISION to "",
                        FEATURE_EXTRACTOR_PATH to "quic_openai_whisper-small.en"
                    ),
            )
    }

    /**
     * Downloads model files for a specific variant and returns a flow of download progress.
     *
     * For OpenAI models (whisperkit-litert/openai_*):
     * - Downloads config.json and tokenizer.json from openai/whisper-*
     * - Downloads AudioEncoder.tflite and TextDecoder.tflite from argmaxinc/whisperkit-litert/openai_whisper-*
     * - Downloads MelSpectrogram.tflite from argmaxinc/whisperkit-litert/openai_whisper-*
     *
     * For Qualcomm models (qualcomm/Whisper_*_En):
     * - Downloads config.json and tokenizer.json from openai/whisper-*.en
     * - Downloads WhisperEncoder.tflite and WhisperDecoder.tflite from qualcomm/Whisper-*-En
     *   and renames them to AudioEncoder.tflite and TextDecoder.tflite respectively
     * - Downloads MelSpectrogram.tflite from argmaxinc/whisperkit-litert/quic_openai_whisper-*
     *
     * @param variant The model variant to download. Must be one of:
     *   - [WhisperKit.Builder.OPENAI_TINY_EN]
     *   - [WhisperKit.Builder.OPENAI_BASE_EN]
     *   - [WhisperKit.Builder.OPENAI_TINY]
     *   - [WhisperKit.Builder.OPENAI_BASE]
     *   - [WhisperKit.Builder.OPENAI_SMALL_EN]
     *   - [WhisperKit.Builder.QUALCOMM_TINY_EN]
     *   - [WhisperKit.Builder.QUALCOMM_BASE_EN]
     *   - [WhisperKit.Builder.QUALCOMM_SMALL_EN]
     * @param root The directory where files will be downloaded
     * @return A flow of download progress, where progress is averaged across all downloads
     */
    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadModelFiles(
        variant: String,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        val config =
            modelConfigs[variant] ?: throw IllegalArgumentException("Invalid variant: $variant")
        return combine(
            downloadConfig(config, root),
            downloadTokenizer(config, root),
            downloadEncoderDecoder(variant, config, root),
            downloadFeatureExtractor(config, root),
        ) { config, tokenizer, encoderDecoder, featureExtractor ->
            // Each flow contributes 1/4 to the total progress
            HuggingFaceApi.Progress(
                fractionCompleted = (
                    config.fractionCompleted + tokenizer.fractionCompleted +
                        encoderDecoder.fractionCompleted + featureExtractor.fractionCompleted
                    ) / 4.0f,
            )
        }.onCompletion {
            // Clean up model directories after all downloads are complete
            if (!variant.startsWith("qualcomm/")) {
                // For OpenAI models, clean up the model directory
                File(root, config[ENCODER_DECODER_REPO]!!).deleteRecursively()
            }
            // Clean up feature extractor directory
            File(root, config[FEATURE_EXTRACTOR_PATH]!!).deleteRecursively()
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadConfig(
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return flow {
            emit(
                huggingFaceApi.getFileMetadata(
                    from = Repo(config[CONFIG_REPO]!!, RepoType.MODELS),
                    filename = "config.json",
                ),
            )
        }.flatMapLatest { tokenizerMetadata ->
            val cachedTokenizerFile = File(root, "config.json")
            if (cachedTokenizerFile.exists() && cachedTokenizerFile.length() == tokenizerMetadata.size) {
                flowOf(HuggingFaceApi.Progress(1.0f))
            } else {
                huggingFaceApi.snapshot(
                    from = Repo(config[CONFIG_REPO]!!, RepoType.MODELS),
                    globFilters = listOf("config.json"),
                    baseDir = root,
                )
            }
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadTokenizer(
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return flow {
            emit(
                huggingFaceApi.getFileMetadata(
                    from = Repo(config[TOKENIZER_REPO]!!, RepoType.MODELS),
                    filename = "tokenizer.json",
                ),
            )
        }.flatMapLatest { tokenizerMetadata ->
            val cachedTokenizerFile = File(root, "tokenizer.json")
            if (cachedTokenizerFile.exists() && cachedTokenizerFile.length() == tokenizerMetadata.size) {
                flowOf(HuggingFaceApi.Progress(1.0f))
            } else {
                huggingFaceApi.snapshot(
                    from = Repo(config[TOKENIZER_REPO]!!, RepoType.MODELS),
                    globFilters = listOf("tokenizer.json"),
                    baseDir = root,
                )
            }
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadEncoderDecoder(
        variant: String,
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return if (variant.startsWith("qualcomm/")) {
            downloadQualcommEncoderDecoder(config, root)
        } else {
            downloadArgmaxEncoderDecoder(config, root)
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadQualcommEncoderDecoder(
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return flow {
            emit(
                huggingFaceApi.getFileMetadata(
                    from = Repo(config[ENCODER_DECODER_REPO]!!, RepoType.MODELS, config[ENCODER_DECODER_REVISION]!!),
                    globFilters = listOf("WhisperEncoder.tflite", "WhisperDecoder.tflite"),
                ),
            )
        }.flatMapLatest { metadataList ->
            val cachedEncoderFile = File(root, "AudioEncoder.tflite")
            val cachedDecoderFile = File(root, "TextDecoder.tflite")
            val encoderMetadata = metadataList.find { it.filename == "WhisperEncoder.tflite" }
            val decoderMetadata = metadataList.find { it.filename == "WhisperDecoder.tflite" }

            if (cachedEncoderFile.exists() && cachedEncoderFile.length() == encoderMetadata?.size &&
                cachedDecoderFile.exists() && cachedDecoderFile.length() == decoderMetadata?.size
            ) {
                flowOf(HuggingFaceApi.Progress(1.0f))
            } else {
                huggingFaceApi.snapshot(
                    from = Repo(config[ENCODER_DECODER_REPO]!!, RepoType.MODELS, config[ENCODER_DECODER_REVISION]!!),
                    globFilters = listOf("WhisperEncoder.tflite", "WhisperDecoder.tflite"),
                    baseDir = root,
                ).onCompletion {
                    File(root, "WhisperEncoder.tflite").renameTo(File(root, "AudioEncoder.tflite"))
                    File(root, "WhisperDecoder.tflite").renameTo(File(root, "TextDecoder.tflite"))
                }
            }
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadArgmaxEncoderDecoder(
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        val modelDir = config[ENCODER_DECODER_REPO]!!
        return flow {
            emit(
                huggingFaceApi.getFileMetadata(
                    from = Repo("argmaxinc/whisperkit-litert", RepoType.MODELS),
                    globFilters =
                    listOf(
                        "$modelDir/AudioEncoder.tflite",
                        "$modelDir/TextDecoder.tflite",
                    ),
                ),
            )
        }.flatMapLatest { metadataList ->
            val cachedEncoderFile = File(root, "AudioEncoder.tflite")
            val cachedDecoderFile = File(root, "TextDecoder.tflite")
            val encoderMetadata =
                metadataList.find { it.filename == "$modelDir/AudioEncoder.tflite" }
            val decoderMetadata =
                metadataList.find { it.filename == "$modelDir/TextDecoder.tflite" }

            if (cachedEncoderFile.exists() && cachedEncoderFile.length() == encoderMetadata?.size &&
                cachedDecoderFile.exists() && cachedDecoderFile.length() == decoderMetadata?.size
            ) {
                flowOf(HuggingFaceApi.Progress(1.0f))
            } else {
                huggingFaceApi.snapshot(
                    from = Repo("argmaxinc/whisperkit-litert", RepoType.MODELS),
                    globFilters =
                    listOf(
                        "$modelDir/AudioEncoder.tflite",
                        "$modelDir/TextDecoder.tflite",
                    ),
                    baseDir = root,
                ).onCompletion {
                    listOf("AudioEncoder.tflite", "TextDecoder.tflite").forEach { fileName ->
                        File(root, "$modelDir/$fileName").renameTo(File(root, fileName))
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun downloadFeatureExtractor(
        config: Map<String, String>,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return flow {
            emit(
                huggingFaceApi.getFileMetadata(
                    from = Repo("argmaxinc/whisperkit-litert", RepoType.MODELS),
                    filename = "${config[FEATURE_EXTRACTOR_PATH]!!}/MelSpectrogram.tflite",
                ),
            )
        }.flatMapLatest { metadata ->
            val cachedFeatureFile = File(root, "MelSpectrogram.tflite")
            if (cachedFeatureFile.exists() && cachedFeatureFile.length() == metadata.size) {
                flowOf(HuggingFaceApi.Progress(1.0f))
            } else {
                huggingFaceApi.snapshot(
                    from = Repo("argmaxinc/whisperkit-litert", RepoType.MODELS),
                    globFilters = listOf("${config[FEATURE_EXTRACTOR_PATH]!!}/MelSpectrogram.tflite"),
                    baseDir = root,
                ).onCompletion {
                    val modelDir = config[FEATURE_EXTRACTOR_PATH]!!
                    File(root, "$modelDir/MelSpectrogram.tflite").renameTo(
                        File(
                            root,
                            "MelSpectrogram.tflite",
                        ),
                    )
                }
            }
        }
    }

    override fun download(
        model: ArgmaxModel,
        variant: String,
        root: File,
    ): Flow<HuggingFaceApi.Progress> {
        return when (model) {
            WHISPER -> downloadModelFiles(variant, root)
        }
    }
}
