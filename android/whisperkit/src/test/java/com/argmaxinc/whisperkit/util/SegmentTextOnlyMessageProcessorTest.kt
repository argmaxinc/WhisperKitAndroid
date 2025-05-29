package com.argmaxinc.whisperkit.util

import com.argmaxinc.whisperkit.TranscriptionSegment
import org.junit.Assert.assertEquals
import org.junit.Test

class SegmentTextOnlyMessageProcessorTest {
    private val processor = SegmentTextOnlyMessageProcessor()

    @Test
    fun `process extracts text segments correctly`() {
        // Given
        val rawMsg = """
            <|startoftranscript|><|0.00|> When I was 27 years old, I left a very demanding job in management consulting.<|18.48|><|18.48|> For a job that was even more demanding, teaching.<|21.84|><|endoftext|>
            <|startoftranscript|><|0.00|> I went to teach seventh graders, Math, in the New York City Public Schools.<|5.88|><|5.88|> And like any teacher, I made quizzes and tests.<|8.08|><|endoftext|>
        """.trimIndent()

        // When
        val result = processor.process(rawMsg)

        // Then
        val expectedSegments = listOf(
            TranscriptionSegment(" When I was 27 years old, I left a very demanding job in management consulting."),
            TranscriptionSegment(" For a job that was even more demanding, teaching."),
            TranscriptionSegment(" I went to teach seventh graders, Math, in the New York City Public Schools."),
            TranscriptionSegment(" And like any teacher, I made quizzes and tests."),
        )

        assertEquals(expectedSegments, result.segments)
    }

    @Test
    fun `process handles multiple segments with same timestamps`() {
        // Given
        val rawMsg = """
            <|startoftranscript|><|0.00|> not fixed, that it can change with your effort.<|4.48|><|4.48|> Dr. Dweck has shown that when kids read and learn<|7.44|><|7.44|> about the brain and how it changes and grows<|10.72|><|10.72|> in response to challenge, they're much more likely<|13.64|><|13.64|> to persevere when they fail because they don't believe<|18.80|><|18.80|> that failure is a permanent condition.<|21.56|><|endoftext|>
            <|startoftranscript|><|0.00|> So, growth mindset is a great idea for building grit, but we need more.<|6.60|><|6.60|> And that's where I'm going to end my remarks, because that's where we are.<|9.28|><|endoftext|>
        """.trimIndent()

        // When
        val result = processor.process(rawMsg)

        // Then
        val expectedSegments = listOf(
            TranscriptionSegment(" not fixed, that it can change with your effort."),
            TranscriptionSegment(" Dr. Dweck has shown that when kids read and learn"),
            TranscriptionSegment(" about the brain and how it changes and grows"),
            TranscriptionSegment(" in response to challenge, they're much more likely"),
            TranscriptionSegment(" to persevere when they fail because they don't believe"),
            TranscriptionSegment(" that failure is a permanent condition."),
            TranscriptionSegment(" So, growth mindset is a great idea for building grit, but we need more."),
            TranscriptionSegment(" And that's where I'm going to end my remarks, because that's where we are."),
        )

        assertEquals(expectedSegments, result.segments)
    }

    @Test
    fun `process handles special characters and short segments`() {
        // Given
        val rawMsg = "<|startoftranscript|><|0.00|> ♪♪<|2.00|><|endoftext|>"

        // When
        val result = processor.process(rawMsg)

        // Then
        val expectedSegments = listOf(
            TranscriptionSegment(" ♪♪"),
        )

        assertEquals(expectedSegments, result.segments)
    }

    @Test
    fun `process handles empty input`() {
        // Given
        val rawMsg = ""

        // When
        val result = processor.process(rawMsg)

        // Then
        assertEquals(emptyList<TranscriptionSegment>(), result.segments)
    }

    @Test
    fun `process handles input with no segments`() {
        // Given
        val rawMsg = "<|startoftranscript|><|endoftext|>"

        // When
        val result = processor.process(rawMsg)

        // Then
        assertEquals(emptyList<TranscriptionSegment>(), result.segments)
    }
}
