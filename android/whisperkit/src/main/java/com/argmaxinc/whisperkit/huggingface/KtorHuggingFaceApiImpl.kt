//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi.FileMetadata
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi.Progress
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.head
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.bodyAsChannel
import io.ktor.utils.io.readRemaining
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.io.readByteArray
import java.io.File

internal class KtorHuggingFaceApiImpl(
    private val config: HuggingFaceApiConfig = HuggingFaceApiConfig(),
    private val client: KtorHuggingFaceClient =
        KtorHuggingFaceClient(
            authToken = config.bearerToken,
        ),
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : HuggingFaceApi {
    private val logger = config.logger

    /**
     * Downloaded file size doesn't match its size from header, need to redownload.
     * This occurs when server close connections prematurely without throwing any exception, often
     * happens when a large file is being downloaded.
     */
    private class FileSizeMismatchException(
        file: String,
        expectedSize: Long,
        actualSize: Long,
    ) : Exception("File size mismatch for $file: expected $expectedSize bytes but got $actualSize bytes")

    private suspend fun getHuggingFaceModel(url: String): ModelInfo {
        return client.httpClient.get(url) {
            url {
                parameters.append("full", "true")
            }
        }.body()
    }

    override suspend fun getFileNames(
        from: Repo,
        globFilters: List<String>,
    ): List<String> {
        return getModelInfo(from).fileNames(globFilters)
    }

    override suspend fun getModelInfo(from: Repo): ModelInfo {
        require(from.type == RepoType.MODELS) {
            "$from needs to have type RepoType.MODELS"
        }
        var url = "/api/${from.type.typeName}/${from.id}"
        if (from.revision != "") {
            url += "/revision/${from.revision}"
        }
        logger.info("Calling HF API at url '${url}'")
        val result = getHuggingFaceModel(url)
        logger.info("Got model info: $result")
        return result
    }

    override suspend fun getFileMetadata(
        from: Repo,
        filename: String,
    ): FileMetadata {
        var revision : String
        if (from.revision == "") { 
            revision = "main" 
        } else { 
            revision = from.revision 
        }
        val response = client.httpClient.head("/${from.id}/resolve/$revision/$filename")
        val size =
            response.headers["X-Linked-Size"]?.toLongOrNull()
                ?: response.headers["Content-Length"]?.toLongOrNull() ?: 0L
        return FileMetadata(
            size = size,
            filename = filename,
            etag = response.headers["ETag"],
            location = response.headers["Location"],
        )
    }

    override suspend fun getFileMetadata(
        from: Repo,
        globFilters: List<String>,
    ): List<FileMetadata> {
        val files = getFileNames(from, globFilters)
        return files.map { filename ->
            getFileMetadata(from, filename)
        }
    }

    /**
     * Downloads files from a HuggingFace repository to a local directory.
     * Progress is reported through a Flow of [Progress] objects.
     *
     * Implementation details:
     * 1. Lists all files in the repository that match the glob patterns
     * 2. Calculates total size by fetching Content-Length headers for each file
     * 3. Downloads each file with retry mechanism
     * 4. Reports progress as a combination of:
     *    - File count progress (files processed / total files)
     *    - Bytes progress (bytes downloaded / total bytes)
     *
     * Note: This function does not handle caching. Use [getFileMetadata] or [getFileMetadata]
     * to implement caching strategies.
     *
     * @param from The repository to download from
     * @param globFilters List of glob patterns to filter which files to download
     * @param baseDir The local directory where files will be downloaded
     * @return Flow of [Progress] objects indicating download progress
     * @throws IllegalStateException if a file download fails after the maximum number of retry attempts
     */
    override fun snapshot(
        from: Repo,
        globFilters: List<String>,
        baseDir: File,
    ): Flow<Progress> {
        return flow {
            baseDir.mkdirs()
            getFileNames(from, globFilters).let { filesToDownload ->
                if (filesToDownload.isEmpty()) {
                    logger.info("No files to download, finish immediately, for Repo(${from.id}, ${from.revision}) and glob filters: $globFilters")
                    emit(Progress(1.0f))
                } else {
                    downloadFilesWithRetry(from, filesToDownload, baseDir)
                }
            }
        }.flowOn(ioDispatcher)
    }

    private suspend fun FlowCollector<Progress>.downloadFilesWithRetry(
        from: Repo,
        files: List<String>,
        baseDir: File,
    ) {
        val totalFiles = files.size.toLong()
        // Get total bytes upfront
        var totalBytes = 0L
        val fileSizes = mutableMapOf<String, Long>()
        files.forEach { file ->
            val metadata = getFileMetadata(from, file)
            fileSizes[file] = metadata.size
            totalBytes += metadata.size
        }
        logger.info("files to download: $files, totalBytes: $totalBytes, totalFiles: ${files.size}")
        for ((fileName, size) in fileSizes) {
            logger.info("  $fileName - $size")
        }
        var downloadedBytes = 0L
        var lastReportedProgress = -1.0f
        files.mapIndexed { index, file ->
            logger.info("checking if $file needs to be downloaded...")
            val targetFile = File(baseDir, file)
            targetFile.parentFile?.mkdirs()
            var retryCount = 0
            var revision : String
            if (from.revision == "") { 
                revision = "main" 
            } else { 
                revision = from.revision 
            }
            val url = "/${from.id}/resolve/$revision/$file"
            while (true) {
                try {
                    logger.info("Retry attempt $retryCount for $file")
                    client.httpClient.prepareGet(url)
                        .execute { response ->
                            val channel = response.bodyAsChannel()
                            targetFile.outputStream().use { output ->
                                while (!channel.isClosedForRead) {
                                    val packet = channel.readRemaining(DEFAULT_BUFFER_SIZE.toLong())
                                    val bytes = packet.readByteArray()
                                    if (bytes.isNotEmpty()) {
                                        output.write(bytes)
                                        downloadedBytes += bytes.size
                                        lastReportedProgress =
                                            emitProgress(
                                                index,
                                                totalFiles,
                                                downloadedBytes,
                                                totalBytes,
                                                lastReportedProgress,
                                            )
                                    }
                                }
                            }
                        }
                    // Verify file size after successful download
                    val actualFileSize = targetFile.length()
                    logger.info(
                        "$file downloaded, size is $actualFileSize, " +
                            "the header report size is ${fileSizes[file]}",
                    )
                    if (actualFileSize != fileSizes[file]) {
                        throw FileSizeMismatchException(
                            file = file,
                            expectedSize = fileSizes[file] ?: 0,
                            actualSize = actualFileSize,
                        )
                    }
                    break // Exit the retry loop on success
                } catch (e: Exception) {
                    logger.error(e, "Error during download for $file (attempt $retryCount)")
                    retryCount++
                    // Clean up partial download before retry
                    if (targetFile.exists()) {
                        // Subtract the failed download bytes from total
                        downloadedBytes -= targetFile.length()
                        targetFile.delete()
                    }
                    if (retryCount >= config.retryCount) {
                        logger.error("Failed to download $file after ${config.retryCount} attempts")
                        throw IllegalStateException(
                            "Failed to download $file after ${config.retryCount} attempts",
                            e,
                        )
                    }
                    logger.info("Wait and retry downloading $file...")
                    // Wait before retrying (exponential backoff)
                    delay(1000L * (1 shl retryCount))
                }
            }
        }
        logger.info("ALL Files downloaded! $files")
    }

    private suspend fun FlowCollector<Progress>.emitProgress(
        index: Int,
        totalFiles: Long,
        downloadedBytes: Long,
        totalBytes: Long,
        lastReportedProgress: Float,
    ): Float {
        val fileProgress = (index + 1).toFloat() / totalFiles
        val bytesProgress = downloadedBytes.toFloat() / totalBytes
        val combinedProgress = (fileProgress + bytesProgress) / 2
        val currentProgressPercent = combinedProgress * 100
        if ((currentProgressPercent - lastReportedProgress >= 1.0f) || (combinedProgress == 1.0f)) {
            // Never report progress lower than what we've already reported
            val progressToReport = maxOf(currentProgressPercent, lastReportedProgress)
            logger.info(
                "reporting progress: fileProgress: $fileProgress, " +
                    "bytesProgress: $bytesProgress, combinedProgress: $combinedProgress",
            )
            emit(
                Progress(
                    fractionCompleted = combinedProgress,
                ),
            )
            return progressToReport
        }
        return lastReportedProgress
    }
}
