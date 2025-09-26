//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

import kotlinx.coroutines.flow.Flow
import java.io.File

/**
 * Interface for interacting with the HuggingFace API.
 * This interface provides methods for retrieving model information and downloading files from HuggingFace repositories.
 */
interface HuggingFaceApi {
    /**
     * Represents metadata for a file in a HuggingFace repository.
     */
    data class FileMetadata(
        val size: Long,
        val filename: String,
        val etag: String? = null,
        val location: String? = null,
    )

    /**
     * Represents the progress of a download operation.
     *
     * @property fractionCompleted A value between 0.0 and 1.0 indicating the completion percentage of the operation
     */
    data class Progress(
        val fractionCompleted: Float,
    ) {
        /**
         * Indicates whether the operation has completed.
         * @return true if the operation is complete (fractionCompleted == 1.0), false otherwise
         */
        val isDone: Boolean
            get() = fractionCompleted == 1.0f
    }

    /**
     * Retrieves a list of file names from a HuggingFace repository that match the specified glob patterns.
     *
     * @param from The repository to search in
     * @param revision The revision/branch/commit to use. Defaults to "main"
     * @param globFilters List of glob patterns to filter files. If empty, all files are returned
     * @return List of file names that match the filters
     */
    suspend fun getFileNames(
        from: Repo,
        revision: String = "main",
        globFilters: List<String> = listOf(),
    ): List<String>

    /**
     * Retrieves detailed information about a model from a HuggingFace repository.
     *
     * @param from The repository containing the model, needs to be type [RepoType.MODELS]
     * @param revision The revision/branch/commit to use. Defaults to "main"
     * @return [ModelInfo] object containing model details
     * @throws IllegalArgumentException if the repository type is not [RepoType.MODELS]
     */
    suspend fun getModelInfo(from: Repo, revision: String = "main"): ModelInfo

    /**
     * Retrieves metadata for a specific file from a HuggingFace repository.
     * This is useful for checking file sizes before downloading.
     *
     * @param from The repository containing the file
     * @param revision The revision/branch/commit to use. Defaults to "main"
     * @param filename The name of the file to get metadata for
     * @return FileMetadata object containing file information
     */
    suspend fun getFileMetadata(
        from: Repo,
        revision: String = "main",
        filename: String,
    ): FileMetadata

    /**
     * Retrieves metadata for multiple files from a HuggingFace repository that match the specified glob patterns.
     * This is useful for checking file sizes before downloading multiple files.
     *
     * @param from The repository containing the files
     * @param revision The revision/branch/commit to use. Defaults to "main"
     * @param globFilters List of glob patterns to filter files. If empty, all files are returned
     * @return List of FileMetadata objects for files that match the filters
     */
    suspend fun getFileMetadata(
        from: Repo,
        revision: String = "main",
        globFilters: List<String> = listOf(),
    ): List<FileMetadata>

    /**
     * Downloads files from a HuggingFace repository to a local directory.
     * Files are only downloaded if they don't exist locally or if their size doesn't match the expected size.
     * Progress is reported through a Flow of [Progress] objects.
     *
     * @param from The repository to download from
     * @param revision The revision/branch/commit to use. Defaults to "main"
     * @param globFilters List of glob patterns to filter which files to download
     * @param baseDir The local directory where files will be downloaded
     * @return Flow of [Progress] objects indicating download progress
     * @throws IllegalStateException if a file download fails after the maximum number of retry attempts
     */
    fun snapshot(
        from: Repo,
        revision: String = "main",
        globFilters: List<String>,
        baseDir: File,
    ): Flow<Progress>
}
