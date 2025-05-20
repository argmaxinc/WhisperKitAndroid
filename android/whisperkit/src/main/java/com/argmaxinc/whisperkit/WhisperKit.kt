//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit

import android.content.Context
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi
import kotlinx.coroutines.flow.Flow

/**
 * WhisperKit is a speech recognition library that provides real-time transcription capabilities.
 * It supports both OpenAI and Qualcomm Whisper models, with various size options and compute backend configurations.
 *
 * !!! EXPERIMENTAL API !!!
 * This API is marked as experimental and may change in future releases.
 * To use this API, you must explicitly opt-in using the @OptIn annotation:
 * ```kotlin
 * @OptIn(ExperimentalWhisperKit::class)
 * fun myFunction() {
 *     val whisperKit = WhisperKit.Builder()
 *         .setModel(WhisperKit.OPENAI_TINY_EN)
 *         .setContext(context)
 *         .setCallback { what, timestamp, msg ->
 *             // Handle transcription output
 *         }
 *         .build()
 * }
 * ```
 * Use with caution in production code.
 *
 * Usage:
 * 1. Create an instance using the Builder pattern:
 *    ```kotlin
 *    val whisperKit = WhisperKit.Builder()
 *        .setModel(WhisperKit.OPENAI_TINY_EN)
 *        .setContext(context)
 *        .setCallback { what, timestamp, msg ->
 *            // Handle transcription output
 *        }
 *        .build()
 *    ```
 *
 * 2. Download and load the model:
 *    ```kotlin
 *    whisperKit.loadModel().collect { progress ->
 *        // Handle download progress
 *    }
 *    ```
 *
 * 3. Initialize the model with audio parameters:
 *    ```kotlin
 *    whisperKit.init(frequency = 16000, channels = 1, duration = 0)
 *    ```
 *
 * 4. Transcribe audio data in chunks:
 *    ```kotlin
 *    val result = whisperKit.transcribe(pcmData)
 *    ```
 *
 * 5. Clean up resources when done:
 *    ```kotlin
 *    whisperKit.deinitialize()
 *    ```
 */
@ExperimentalWhisperKit
interface WhisperKit {
    /**
     * Downloads and loads the specified model variant into memory.
     * This must be called before any transcription can occur.
     *
     * @return A Flow that emits download progress updates
     * @throws WhisperKitException if model loading fails
     */
    @Throws(WhisperKitException::class)
    suspend fun loadModel(): Flow<HuggingFaceApi.Progress>

    /**
     * Initializes the model with the specified audio parameters.
     * Must be called after loadModel() and before any transcription.
     *
     * @param frequency The sample rate of the audio (typically 16000 Hz)
     * @param channels The number of audio channels (typically 1 for mono)
     * @param duration The duration of the audio in milliseconds (0 for streaming)
     * @throws WhisperKitException if initialization fails
     */
    @Throws(WhisperKitException::class)
    suspend fun init(
        frequency: Int,
        channels: Int,
        duration: Long,
    )

    /**
     * Transcribes a chunk of PCM audio data.
     * Can be called multiple times with different chunks of audio data.
     * The transcription results will be delivered through the TextOutputCallback.
     *
     * @param pcmData Raw PCM audio data (16-bit signed integer format)
     * @return Number of seconds transcribed, or negative value on error
     * @throws WhisperKitException if transcription fails or model is not initialized
     */
    @Throws(WhisperKitException::class)
    fun transcribe(pcmData: ByteArray): Int

    /**
     * Cleans up resources and releases the model from memory.
     * Should be called when transcription is complete.
     *
     * @throws WhisperKitException if cleanup fails
     */
    @Throws(WhisperKitException::class)
    fun deinitialize()

    /**
     * Callback interface for receiving transcription results.
     * The callback will be invoked with:
     * - MSG_INIT (0): When init() succeeds, indicating the model is ready for transcription
     * - MSG_TEXT_OUT (1): Contains transcription results for the previous transcribe() call
     * - MSG_CLOSE (2): When deinitialize() succeeds, indicating cleanup is complete
     */
    fun interface TextOutputCallback {
        companion object {
            /**
             * Message types for TextOutputCallback:
             * - MSG_INIT (0): Sent when init() succeeds, indicating the model is ready for transcription
             * - MSG_TEXT_OUT (1): Contains transcription results for the previous transcribe() call
             * - MSG_CLOSE (2): Sent when deinitialize() succeeds, indicating cleanup is complete
             */
            const val MSG_INIT = 0
            const val MSG_TEXT_OUT = 1
            const val MSG_CLOSE = 2
        }

        /**
         * Called when new transcription output is available.
         *
         * @param what The message type:
         *             - MSG_INIT (0): init() succeeded, model is ready
         *             - MSG_TEXT_OUT (1): transcription results from previous transcribe() call
         *             - MSG_CLOSE (2): deinitialize() succeeded, cleanup complete
         * @param timestamp The timestamp of the transcribed segment (only valid for MSG_TEXT_OUT)
         * @param msg The transcribed text or status message:
         *           - For MSG_INIT: initialization status
         *           - For MSG_TEXT_OUT: transcribed text
         *           - For MSG_CLOSE: cleanup status
         */
        fun onTextOutput(
            what: Int,
            timestamp: Float,
            msg: String,
        )
    }

    /**
     * Builder class for creating WhisperKit instances.
     * All required parameters must be set before calling build().
     */
    class Builder {
        companion object {
            // Model variants
            const val OPENAI_TINY_EN = "whisperkit-litert/openai_whisper-tiny.en"
            const val OPENAI_BASE_EN = "whisperkit-litert/openai_whisper-base.en"
            const val OPENAI_SMALL_EN = "whisperkit-litert/openai_whisper-small.en"
            const val QUALCOMM_TINY_EN = "qualcomm/Whisper_Tiny_En"
            const val QUALCOMM_BASE_EN = "qualcomm/Whisper_Base_En"
            const val QUALCOMM_SMALL_EN = "qualcomm/Whisper_Small_En"

            // Compute units used for encoder/decoder backend
            const val CPU_ONLY = 1
            const val CPU_AND_GPU = 2
            const val CPU_AND_NPU = 3
        }

        private var model: String? = null
        private var context: Context? = null
        private var encoderBackend: Int = CPU_AND_NPU
        private var decoderBackend: Int = CPU_AND_NPU
        private var callback: TextOutputCallback? = null

        /**
         * Sets the model variant to use for transcription.
         * Must be one of the predefined model constants.
         *
         * @throws WhisperKitException if model is not one of the predefined variants
         */
        @Throws(WhisperKitException::class)
        fun setModel(model: String): Builder {
            if (model !in
                listOf(
                    OPENAI_TINY_EN,
                    OPENAI_BASE_EN,
                    OPENAI_SMALL_EN,
                    QUALCOMM_TINY_EN,
                    QUALCOMM_BASE_EN,
                    QUALCOMM_SMALL_EN,
                )
            ) {
                throw WhisperKitException("Model must be one of the predefined variants")
            }
            this.model = model
            return this
        }

        /**
         * Sets the Android Application Context required for model loading and file operations.
         * Always use applicationContext to avoid memory leaks and ensure proper lifecycle management.
         * Example:
         * ```kotlin
         * .setApplicationContext(context.applicationContext)
         * ```
         */
        fun setApplicationContext(applicationContext: Context): Builder {
            this.context = applicationContext
            return this
        }

        /**
         * Sets the compute backend for the encoder.
         * Must be one of: [CPU_ONLY], [CPU_AND_GPU], or [CPU_AND_NPU]
         *
         * @throws WhisperKitException if backend is invalid
         */
        @Throws(WhisperKitException::class)
        fun setEncoderBackend(backend: Int): Builder {
            if (backend !in listOf(CPU_ONLY, CPU_AND_GPU, CPU_AND_NPU)) {
                throw WhisperKitException("Invalid encoder backend")
            }
            this.encoderBackend = backend
            return this
        }

        /**
         * Sets the compute backend for the decoder.
         * Must be one of: [CPU_ONLY], [CPU_AND_GPU], or [CPU_AND_NPU]
         *
         * @throws WhisperKitException if backend is invalid
         */
        @Throws(WhisperKitException::class)
        fun setDecoderBackend(backend: Int): Builder {
            if (backend !in listOf(CPU_ONLY, CPU_AND_GPU, CPU_AND_NPU)) {
                throw WhisperKitException("Invalid decoder backend")
            }
            this.decoderBackend = backend
            return this
        }

        /**
         * Sets the callback for receiving transcription results.
         */
        fun setCallback(callback: TextOutputCallback): Builder {
            this.callback = callback
            return this
        }

        /**
         * Creates a new WhisperKit instance with the configured parameters.
         * @throws WhisperKitException if any required parameters are missing
         */
        @Throws(WhisperKitException::class)
        fun build(): WhisperKit {
            val model = requireNotNull(model) { throw WhisperKitException("Model must be set") }
            val context =
                requireNotNull(context) { throw WhisperKitException("Context must be set") }
            val callback =
                requireNotNull(callback) { throw WhisperKitException("Callback must be set") }
            return WhisperKitImpl(
                context = context,
                model = model,
                encoderBackend = encoderBackend,
                decoderBackend = decoderBackend,
                callback = callback,
            )
        }
    }
}
