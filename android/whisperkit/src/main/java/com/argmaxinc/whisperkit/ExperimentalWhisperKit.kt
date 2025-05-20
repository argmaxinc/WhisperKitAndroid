package com.argmaxinc.whisperkit

import kotlin.RequiresOptIn
import kotlin.annotation.AnnotationTarget

/**
 * Marks the WhisperKit API as experimental.
 * This annotation indicates that the API is still in development and may change in future releases.
 */
@RequiresOptIn(
    level = RequiresOptIn.Level.WARNING,
    message = "This API is experimental and may change in future releases. Use with caution in production code.",
)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
annotation class ExperimentalWhisperKit
