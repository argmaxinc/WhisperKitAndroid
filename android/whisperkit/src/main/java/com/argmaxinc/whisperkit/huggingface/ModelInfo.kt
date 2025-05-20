//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.nio.file.FileSystems
import java.nio.file.Paths

/**
 * Represents detailed information about a model from the HuggingFace repository.
 * This class is used to deserialize the JSON response from the HuggingFace API.
 *
 * @property id The unique identifier of the model
 * @property modelId The model's identifier
 * @property private Whether the model is private
 * @property pipelineTag The type of pipeline this model is designed for
 * @property libraryName The name of the library this model is built with
 * @property tags List of tags associated with the model
 * @property downloads Number of times the model has been downloaded
 * @property likes Number of likes the model has received
 * @property author The author of the model
 * @property sha The Git commit SHA of the model
 * @property lastModified Timestamp of the last modification
 * @property gated Whether the model requires authentication to access
 * @property disabled Whether the model is disabled
 * @property modelIndex The model index identifier
 * @property config Configuration parameters for the model
 * @property cardData Additional metadata about the model
 * @property siblings List of files associated with the model
 * @property spaces List of associated HuggingFace Spaces
 * @property createdAt Timestamp when the model was created
 * @property usedStorage Amount of storage used by the model in bytes
 */
@Serializable
data class ModelInfo(
    @SerialName("_id") val id: String? = null,
    val modelId: String? = null,
    val siblings: List<Sibling>? = null,
) {
    /**
     * Filters the model's files based on glob patterns.
     *
     * @param globFilters List of glob patterns to filter files. If empty, no files are returned
     * @return List of filenames that match the glob patterns
     */
    fun fileNames(globFilters: List<String> = emptyList()): List<String> {
        return siblings?.mapNotNull { it.rfilename }?.filter { filename ->
            globFilters.any { pattern ->
                val matcher = FileSystems.getDefault().getPathMatcher("glob:$pattern")
                matcher.matches(Paths.get(filename))
            }
        } ?: emptyList()
    }

    /**
     * Returns a list of unique first level directory names from the model's files.
     * This is useful for identifying different model variants or configurations.
     *
     * @return List of unique directory names, sorted alphabetically
     */
    fun dirNames(): List<String> {
        return fileNames().filter { it.contains("/") }.map { it.split("/").first() }.distinct()
            .sorted()
    }
}

/**
 * Additional metadata about a model from its model card.
 *
 * @property prettyName A human-readable name for the model
 * @property viewer Whether the model has a viewer
 * @property libraryName The name of the library this model is built with
 * @property tags List of tags associated with the model
 */
@Serializable
data class CardData(
    @SerialName("pretty_name") val prettyName: String? = null,
    val viewer: Boolean? = null,
    @SerialName("library_name") val libraryName: String? = null,
    val tags: List<String>? = null,
)

/**
 * Represents a file associated with a model.
 *
 * @property rfilename The relative filename of the file in the repository
 */
@Serializable
data class Sibling(
    val rfilename: String? = null,
)
