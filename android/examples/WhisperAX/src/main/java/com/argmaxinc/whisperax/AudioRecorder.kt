//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

class AudioRecorder(private val context: Context) {

    companion object {
        private const val TAG = "AudioRecorder"
        const val SAMPLE_RATE = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val BUFFER_SIZE_FACTOR = 2
    }

    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)

    private val _bufferEnergy = MutableStateFlow<List<Float>>(emptyList())
    val bufferEnergy: StateFlow<List<Float>> = _bufferEnergy

    private val energyValues = mutableListOf<Float>()
    private val pcmBuffer = java.io.ByteArrayOutputStream()
    private var recordingThread: Thread? = null

    @RequiresPermission(android.Manifest.permission.RECORD_AUDIO)
    fun startRecording(outputFile: File): Boolean {
        if (isRecording.get()) {
            Log.w(TAG, "Already recording")
            return false
        }

        try {
            val minBufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
            )

            if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
                Log.e(TAG, "Invalid buffer size")
                return false
            }

            val bufferSize = minBufferSize * BUFFER_SIZE_FACTOR

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize,
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord not initialized")
                return false
            }

            energyValues.clear()
            isRecording.set(true)
            pcmBuffer.reset()
            audioRecord?.startRecording()

            recordingThread = Thread {
                writeAudioDataToFile(outputFile, bufferSize)
            }
            recordingThread?.start()

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            releaseResources()
            return false
        }
    }

    fun stopRecording() {
        if (!isRecording.get()) {
            return
        }

        isRecording.set(false)

        try {
            recordingThread?.join(1000)
            releaseResources()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
        }
    }

    private fun writeAudioDataToFile(outputFile: File, bufferSize: Int) {
        try {
            val outputStream = FileOutputStream(outputFile)

            writeWavHeader(outputStream, SAMPLE_RATE, false)

            val audioBuffer = ShortArray(bufferSize / 2)
            var totalBytesWritten = 0

            while (isRecording.get()) {
                val readResult = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: -1

                if (readResult > 0) {
                    val energy = calculateEnergy(audioBuffer, readResult)
                    energyValues.add(energy)

                    if (energyValues.size > 300) {
                        energyValues.removeAt(0)
                    }

                    _bufferEnergy.value = energyValues.toList()

                    for (i in 0 until readResult) {
                        val value = audioBuffer[i]
                        outputStream.write(value.toByte().toInt())
                        outputStream.write((value.toInt() shr 8).toByte().toInt())
                        totalBytesWritten += 2
                        pcmBuffer.write(value.toByte().toInt())
                        pcmBuffer.write((value.toInt() shr 8).toByte().toInt())
                    }
                }
            }

            outputStream.close()
            updateWavHeader(outputFile, totalBytesWritten)
        } catch (e: IOException) {
            Log.e(TAG, "Error processing audio data", e)
        }
    }

    private fun calculateEnergy(buffer: ShortArray, length: Int): Float {
        var sum = 0.0f

        for (i in 0 until length) {
            val sample = abs(buffer[i].toFloat() / 32768.0f)
            sum += sample * sample
        }

        val rms = if (length > 0) Math.sqrt(sum / length.toDouble()).toFloat() else 0.0f

        return minOf(1.0f, rms * 5.0f)
    }

    private fun releaseResources() {
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing resources", e)
        }
    }

    private fun writeWavHeader(outputStream: FileOutputStream, sampleRate: Int, isStereo: Boolean) {
        try {
            outputStream.write("RIFF".toByteArray())
            outputStream.write(ByteArray(4))
            outputStream.write("WAVE".toByteArray())

            outputStream.write("fmt ".toByteArray())
            val subchunk1Size = 16
            outputStream.write(intToByteArray(subchunk1Size))
            val audioFormat = 1
            outputStream.write(shortToByteArray(audioFormat.toShort()))
            val numChannels = if (isStereo) 2 else 1
            outputStream.write(shortToByteArray(numChannels.toShort()))
            outputStream.write(intToByteArray(sampleRate))
            val byteRate = sampleRate * numChannels * 2
            outputStream.write(intToByteArray(byteRate))
            val blockAlign = numChannels * 2
            outputStream.write(shortToByteArray(blockAlign.toShort()))
            val bitsPerSample = 16
            outputStream.write(shortToByteArray(bitsPerSample.toShort()))

            outputStream.write("data".toByteArray())
            outputStream.write(ByteArray(4))
        } catch (e: IOException) {
            Log.e(TAG, "Error writing WAV header", e)
        }
    }

    private fun updateWavHeader(wavFile: File, audioDataSize: Int) {
        try {
            val raf = java.io.RandomAccessFile(wavFile, "rw")

            raf.seek(4L)
            raf.write(intToByteArray(audioDataSize + 36))

            raf.seek(40L)
            raf.write(intToByteArray(audioDataSize))

            raf.close()
        } catch (e: IOException) {
            Log.e(TAG, "Error updating WAV header", e)
        }
    }

    private fun intToByteArray(value: Int): ByteArray {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
    }

    private fun shortToByteArray(value: Short): ByteArray {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array()
    }
    fun getPcmData(): ByteArray = pcmBuffer.toByteArray()
}
