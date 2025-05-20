//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

/**
 * Configuration class for the HuggingFace API implementation.
 * This class holds all configurable parameters for the API client.
 *
 * @property retryCount The maximum number of retry attempts for failed downloads. Defaults to [DEFAULT_MAX_RETRY].
 * @property bearerToken Optional authentication token for accessing private repositories. Defaults to null.
 * @property logger The logger implementation to use for API operations. Defaults to [NoOpHuggingFaceLogger].
 */
data class HuggingFaceApiConfig(
    val retryCount: Int = DEFAULT_MAX_RETRY,
    val bearerToken: String? = null,
    val logger: HuggingFaceLogger = NoOpHuggingFaceLogger,
) {
    companion object {
        /**
         * Default maximum number of retry attempts for failed downloads.
         */
        const val DEFAULT_MAX_RETRY = 3
    }
}
