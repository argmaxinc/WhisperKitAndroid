package com.argmaxinc.whisperkit.util

import com.argmaxinc.whisperkit.TranscriptionResult
import com.argmaxinc.whisperkit.TranscriptionSegment

/**
 * A processor to only extract segment text from raw string, ignoring all timestamps or windows
 */
internal class SegmentTextOnlyMessageProcessor : MessageProcessor {
    private companion object {
        private val TIMESTAMP_PATTERN = "<\\|(\\d+\\.\\d+)\\|>".toRegex()

        // Pattern to match any <|str|> that's not a timestamp
        private val NON_TIMESTAMP_PATTERN = "<\\|(?!\\d+\\.\\d+)[^>]*\\|>".toRegex()
    }

    override fun process(rawMsg: String): TranscriptionResult {
        // Remove any markers that aren't timestamps
        val cleanMsg = rawMsg.replace(NON_TIMESTAMP_PATTERN, "")

        val segments = mutableListOf<TranscriptionSegment>()

        // Find all timestamp markers
        val matches = TIMESTAMP_PATTERN.findAll(cleanMsg).toList()

        for (i in 0 until matches.size - 1) {
            val startMatch = matches[i]
            val endMatch = matches[i + 1]

            // TODD: add start and end to each segment
            // val start = startMatch.groupValues[1].toFloat()
            // val end = endMatch.groupValues[1].toFloat()

            // Extract text between timestamps
            val textStart = startMatch.range.last + 1
            val textEnd = endMatch.range.first
            val text = cleanMsg.substring(textStart, textEnd)

            if (text != "\n" && text.isNotEmpty()) {
                segments.add(TranscriptionSegment(text))
            }
        }

        return TranscriptionResult(text = rawMsg, segments = segments)
    }
}
