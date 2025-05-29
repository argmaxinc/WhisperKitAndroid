package com.argmaxinc.whisperkit.util

import com.argmaxinc.whisperkit.TranscriptionResult

/**
 * Processor to convert raw model output Strings into [TranscriptionResult]
 */
internal interface MessageProcessor {
    fun process(rawMsg: String): TranscriptionResult
}
