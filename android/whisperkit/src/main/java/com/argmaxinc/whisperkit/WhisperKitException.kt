package com.argmaxinc.whisperkit

/**
 * Exception thrown by WhisperKit when an error occurs during model operations.
 * This exception can be thrown during:
 * - Model loading and initialization
 * - Transcription processing
 * - Resource cleanup
 * - Invalid configuration
 *
 * @param message A detailed description of the error
 * @param cause The underlying exception that caused this error, if any
 */
class WhisperKitException(message: String, cause: Throwable? = null) : Exception(message, cause)
