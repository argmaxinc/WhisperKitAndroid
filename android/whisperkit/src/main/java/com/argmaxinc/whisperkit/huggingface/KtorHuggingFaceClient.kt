//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.request.header
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

/**
 * HTTP client for interacting with the HuggingFace API.
 * This class configures a Ktor HTTP client with appropriate settings for the HuggingFace API,
 * including authentication, timeouts, and JSON serialization.
 *
 * @property authToken Optional authentication token for accessing private repositories
 */
internal class KtorHuggingFaceClient(
    authToken: String?,
) {
    private companion object {
        /** Base URL for the HuggingFace API */
        const val BASE_URL = "https://huggingface.co"
    }

    /**
     * JSON serializer configuration for handling HuggingFace API responses.
     * Configured to be lenient and ignore unknown fields to handle API changes gracefully.
     */
    private val json =
        Json {
            ignoreUnknownKeys = true
            isLenient = true
            coerceInputValues = true
        }

    /**
     * Configured HTTP client for making requests to the HuggingFace API.
     * Features:
     * - 1-second socket timeout between packets
     * - JSON content negotiation
     * - Default request configuration with base URL and content type
     * - Optional bearer token authentication
     */
    val httpClient =
        HttpClient(CIO) {
            install(HttpTimeout) {
                socketTimeoutMillis = 1000 // 1 seconds between packets
            }

            install(ContentNegotiation) {
                json(json)
            }

            defaultRequest {
                url(BASE_URL)
                header(HttpHeaders.ContentType, ContentType.Application.Json)
                authToken?.let {
                    header(HttpHeaders.Authorization, "Bearer $it")
                }
            }
        }
}
