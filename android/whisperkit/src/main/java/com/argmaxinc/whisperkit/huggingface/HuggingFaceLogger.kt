//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

/**
 * Interface for logging operations in the HuggingFace API implementation.
 * This interface provides methods for logging informational and error messages,
 * with support for including throwable exceptions in error logs.
 */
interface HuggingFaceLogger {
    fun info(message: String)

    fun error(message: String)

    fun error(
        throwable: Throwable,
        message: String,
    )
}

/**
 * A no-operation implementation of [HuggingFaceLogger] that silently discards all log messages.
 * This implementation is useful when logging is not needed or should be disabled.
 */
object NoOpHuggingFaceLogger : HuggingFaceLogger {
    override fun info(message: String) {
        // No-op
    }

    override fun error(message: String) {
        // No-op
    }

    override fun error(
        throwable: Throwable,
        message: String,
    ) {
        // No-op
    }
}
