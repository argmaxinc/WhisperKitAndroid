//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

package com.argmaxinc.whisperkit.network

import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApi.Progress
import kotlinx.coroutines.flow.Flow
import java.io.File

interface ArgmaxModelDownloader {
    /**
     * Downloads all model files for a specific model variant.
     *
     * @param model The model to download
     * @param variant The specific variant to download
     * @param root The root directory where model files will be downloaded
     * @return A Flow of [HuggingFaceApi.Progress] that reports download progress
     */
    fun download(
        model: ArgmaxModel,
        variant: String,
        root: File,
    ): Flow<Progress>
}
