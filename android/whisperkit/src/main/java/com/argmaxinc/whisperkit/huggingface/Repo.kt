//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.huggingface

/**
 * Represents a HuggingFace repository.
 *
 * @property id The unique identifier of the repository, typically in the format "username/repo-name"
 * @property type The type of repository, which determines its purpose and available operations
 */
data class Repo(
    val id: String,
    val type: RepoType,
)

/**
 * Enumeration of possible HuggingFace repository types.
 * Each type corresponds to a different category of content on the HuggingFace platform.
 *
 * @property typeName The string representation of the repository type as used in the HuggingFace API
 */
enum class RepoType(val typeName: String) {
    /** Repository containing machine learning models */
    MODELS("models"),

    /** Repository containing datasets for training or evaluation */
    DATASETS("datasets"),

    /** Repository containing interactive demos and applications */
    SPACES("spaces"),
}
