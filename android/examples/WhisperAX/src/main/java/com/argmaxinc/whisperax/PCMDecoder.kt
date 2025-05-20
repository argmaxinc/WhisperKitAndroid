//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.io.FileInputStream
import java.io.IOException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class PCMDecoder(clientCallbacks: AudioDecoderCallbacks) {
    private val DECODE_INPUT_SIZE = 524288
    private val TAG = "PCMDecoder"

    private val clientCB = clientCallbacks
    private var lock = ReentrantLock()
    private var isReleased = false

    private lateinit var extractor: MediaExtractor
    private lateinit var codec: MediaCodec
    private lateinit var inputStream: FileInputStream
    private var isEOS: Boolean = false

    fun isEndOfStream(): Boolean {
        return isEOS
    }
    fun decodeFileStart(audioFile: String?) {
        extractor = MediaExtractor()

        try {
            inputStream = FileInputStream(audioFile)
            extractor.setDataSource(inputStream.fd)

            var format: MediaFormat? = null
            var trackIndex = -1

            for (i in 0 until extractor.trackCount) {
                format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME)
                if (mime!!.startsWith("audio/")) {
                    trackIndex = i
                    break
                }
            }

            if (trackIndex < 0) {
                Log.e(TAG, "No audio track found in file.")
                return
            }

            extractor.selectTrack(trackIndex)
            val channels = format!!.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val duration = format.getLong(MediaFormat.KEY_DURATION)
            clientCB.onAudioFormat(sampleRate, channels, duration)

            format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, DECODE_INPUT_SIZE)
            codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec.configure(format, null, null, 0)
            codec.start()
        } catch (e: IOException) {
            Log.e(TAG, "Error decodeFileStart", e)
        }
    }

    fun waitInLoop() {
        val bufferInfo = MediaCodec.BufferInfo()

        while (!isEOS && !isReleased) {
            val inputBufferIndex = codec.dequeueInputBuffer(1000)
            if (inputBufferIndex >= 0) {
                val inputBuffer = codec.getInputBuffer(inputBufferIndex)
                inputBuffer?.clear()

                val sampleSize = extractor.readSampleData(inputBuffer!!, 0)
                if (sampleSize < 0) {
                    codec.queueInputBuffer(
                        inputBufferIndex,
                        0,
                        0,
                        0L,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                    )
                    isEOS = true
                } else {
                    val presentationTimeUs = extractor.sampleTime
                    codec.queueInputBuffer(inputBufferIndex, 0, sampleSize, presentationTimeUs, 0)
                    extractor.advance()
                }
            }

            val outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 1000)
            if (outputBufferIndex >= 0) {
                val outputBuffer = codec.getOutputBuffer(outputBufferIndex)

                val pcmData = ByteArray(bufferInfo.size)
                outputBuffer?.get(pcmData)
                outputBuffer?.clear()

                clientCB.onOutputBuffer(pcmData, extractor.sampleTime)

                if (!isReleased) {
                    codec.releaseOutputBuffer(outputBufferIndex, false)
                }
            } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val newFormat = codec.outputFormat
                Log.d(TAG, "Output format changed: $newFormat")
            }
        }
    }

    fun decodeStopRelease() {
        lock.withLock {
            if (isReleased) {
                Log.d(TAG, "Decoder already released, skipping")
                return
            }

            try {
                isReleased = true
                isEOS = true
                Thread.sleep(100)
                extractor.release()
                codec.stop()
                codec.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error in decodeStopRelease", e)
            } finally {
                try {
                    inputStream.close()
                    Thread.sleep(100)
                    clientCB.onDecodeClose()
                } catch (e: IOException) {
                    Log.e(TAG, "Error closing file input stream", e)
                }
            }
        }
    }
}
