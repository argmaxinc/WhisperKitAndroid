//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit

import android.content.Context
import android.util.Log
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi
import com.argmaxinc.whisperkit.network.ArgmaxModel
import com.argmaxinc.whisperkit.network.ArgmaxModelDownloader
import com.argmaxinc.whisperkit.network.ArgmaxModelDownloaderImpl
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.mapLatest
import java.io.File

@OptIn(ExperimentalWhisperKit::class)
internal class WhisperKitImpl(
    context: Context,
    private val model: String,
    encoderBackend: Int,
    decoderBackend: Int,
    private val callback: WhisperKit.TextOutputCallback,
    private val argmaxModelDownloader: ArgmaxModelDownloader = ArgmaxModelDownloaderImpl(),
) : WhisperKit {
    companion object {
        private const val TAG = "WhisperKitImpl"
        private const val MODEL_DIR = "argmaxinc/models"
        private const val REPORT_DIR = "argmaxinc/reports"
    }

    private var isModelLoaded = false
    private val modelPath: String
    private val loadConfig: String

    init {
        try {
            System.loadLibrary("whisperkit_jni")
            val baseDir = context.filesDir.path
            val modelDir = "$baseDir/$MODEL_DIR"
            val reportDir = "$baseDir/$REPORT_DIR"
            val libDir = context.applicationInfo.nativeLibraryDir
            val cacheDir = context.cacheDir.path
            // Get the model name from the variant
            // (e.g., "openai_whisper-tiny.en" from "whisperkit-litert/openai_whisper-tiny.en")
            val modelName = model.split("/").last()
            modelPath = "$modelDir/$modelName"

            // Create directories if they don't exist
            File(modelDir).mkdirs()
            File(reportDir).mkdirs()
            File(cacheDir).mkdirs()

            // Initialize the loadConfig JSON
            loadConfig =
                """
                {
                    "lib": "$libDir",
                    "cache": "$cacheDir",
                    "size": "dummy",
                    "encoder_backend": $encoderBackend,
                    "decoder_backend": $decoderBackend,
                    "model_path": "$modelPath",
                    "report_path": "$reportDir"
                }
                """.trimIndent()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize WhisperKit: ${e.message}")
            throw WhisperKitException("Failed to initialize WhisperKit: ${e.message}", e)
        }
    }

    override suspend fun init(
        frequency: Int,
        channels: Int,
        duration: Long,
    ) {
        val configJson =
            """
            {
                "freq": $frequency,
                "ch": $channels,
                "dur": $duration,
            }
            """.trimIndent()

        // Initialize pipeline
        val initResult = init(configJson)
        if (initResult != 0) {
            Log.e(TAG, "Failed to initialize audio pipeline: $initResult")
            throw WhisperKitException("Failed to initialize audio pipeline: $initResult")
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    override suspend fun loadModel(): Flow<HuggingFaceApi.Progress> =
        try {
            argmaxModelDownloader.download(
                ArgmaxModel.WHISPER,
                model,
                File(modelPath),
            ).mapLatest { progress ->
                progress
                if (progress.isDone) {
                    // Load models
                    Log.i(TAG, "download complete, load with $loadConfig")
                    val result = loadModels(loadConfig)
                    if (result != 0) {
                        Log.e(TAG, "Failed to load models: JNI returned error code $result")
                        throw WhisperKitException("Failed to load models: JNI returned error code $result")
                    }
                    isModelLoaded = true
                    HuggingFaceApi.Progress(100f)
                } else {
                    // During download, just scale the progress to 95%
                    progress.copy(fractionCompleted = progress.fractionCompleted * 0.95f)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}")
            throw WhisperKitException("Failed to initialize: ${e.message}", e)
        }

    override fun transcribe(pcmData: ByteArray): Int {
        check(isModelLoaded) { throw WhisperKitException("Model not loaded") }
        try {
            return writeData(pcmData)
        } catch (e: Exception) {
            Log.e(TAG, "Error during transcription", e)
            throw WhisperKitException("Error during transcription", e)
        }
    }

    override fun deinitialize() {
        try {
            val closeResult = close()
            if (closeResult != 0) {
                Log.e(TAG, "Failed to close streaming: $closeResult")
                throw WhisperKitException("Failed to close streaming: $closeResult")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Transcription failed: ${e.message}")
            throw WhisperKitException("Transcription failed: ${e.message}", e)
        }
    }

    // Callback from JNI native code
    private fun onTextOutput(
        what: Int,
        timestamp: Float,
        msg: String,
    ) {
        try {
            callback.onTextOutput(what, timestamp, msg)
        } catch (e: Exception) {
            Log.e(TAG, "Callback execution failed: ${e.message}")
            throw WhisperKitException("Callback execution failed: ${e.message}", e)
        }
    }

    // JNI native methods
    external fun loadModels(jsonstr: String): Int

    external fun init(jsonstr: String): Int

    external fun writeData(pcmbuffer: ByteArray): Int

    external fun close(): Int

    // Unused
    external fun setBackend(
        encoderBackend: Int,
        decoderBackend: Int,
    ): Int
}
