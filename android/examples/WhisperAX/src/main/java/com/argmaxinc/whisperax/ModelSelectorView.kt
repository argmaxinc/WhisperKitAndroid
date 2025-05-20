//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.util.Log
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.unit.dp

enum class ModelState(val description: String) {
    UNLOADED("Not Loaded"),
    LOADING("Loading"),
    DOWNLOADING("Downloading"),
    PREWARMING("Specializing"),
    LOADED("Loaded"),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectorView(viewModel: WhisperViewModel) {
    val modelState by viewModel.modelState.collectAsState()
    val availableModels = viewModel.availableModels
    val selectedModel by viewModel.selectedModel.collectAsState()
    val loadingProgress by viewModel.loadingProgress.collectAsState()
    val downloadedModels = viewModel.downloadedModels

    LaunchedEffect(modelState) {
        Log.d("ModelSelectorView", "Model state changed to: ${modelState.description}")
    }

    LaunchedEffect(Unit) {
        viewModel.listModels()
    }

    Column {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth(),
        ) {
            StatusIndicator(modelState)

            Spacer(modifier = Modifier.width(8.dp))

            Text(
                text = modelState.description,
                style = MaterialTheme.typography.bodyMedium,
            )

            Spacer(modifier = Modifier.weight(1f))
        }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        ) {
            var expanded by remember { mutableStateOf(false) }

            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = !expanded },
            ) {
                OutlinedTextField(
                    value = selectedModel.replaceFirstChar { it.uppercase() }
                        .replace("-", " "),
                    onValueChange = {},
                    readOnly = true,
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                )

                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                ) {
                    availableModels.forEach { model ->
                        val isDownloaded = downloadedModels.contains(model)
                        DropdownMenuItem(
                            text = {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    val icon = if (isDownloaded) {
                                        Icons.Default.CheckCircle
                                    } else {
                                        Icons.Default.CloudDownload
                                    }
                                    Icon(
                                        imageVector = icon,
                                        contentDescription = null,
                                        modifier = Modifier.size(16.dp),
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        model.replaceFirstChar { it.uppercase() }
                                            .replace("-", " "),
                                    )
                                }
                            },
                            onClick = {
                                viewModel.selectModel(model)
                                expanded = false
                            },
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.width(8.dp))

            IconButton(
                onClick = { /* Open folder */ },
            ) {
                Icon(
                    imageVector = Icons.Default.Folder,
                    contentDescription = "Open folder",
                )
            }

            IconButton(
                onClick = { /* Open link */ },
            ) {
                Icon(
                    imageVector = Icons.Default.Link,
                    contentDescription = "Open link",
                )
            }
        }
    }

    Spacer(modifier = Modifier.height(8.dp))

    if (modelState == ModelState.UNLOADED) {
        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Button(
            onClick = { viewModel.loadModel(selectedModel) },
            modifier = Modifier.fillMaxWidth().height(48.dp),
        ) {
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = if (downloadedModels.contains(selectedModel)) {
                        Icons.Default.PlayArrow
                    } else {
                        Icons.Default.CloudDownload
                    },
                    contentDescription = null,
                    modifier = Modifier.size(24.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Load Model")
            }
        }
    } else if (modelState == ModelState.PREWARMING) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                LinearProgressIndicator(
                    modifier = Modifier.weight(1f),
                )

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = "Specializing...",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Text(
                text = "Specializing $selectedModel for your device...\nThis can take several minutes on first load.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    } else if (modelState == ModelState.LOADED && loadingProgress != 1.0f) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                LinearProgressIndicator(
                    progress = { 1.0f },
                    modifier = Modifier.weight(1f),
                )

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = "100%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Text(
                text = "Model loaded successfully",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    } else if (loadingProgress < 1.0f) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                LinearProgressIndicator(
                    progress = { loadingProgress },
                    modifier = Modifier.weight(1f),
                )

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = "${(loadingProgress * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (modelState == ModelState.DOWNLOADING) {
                Text(
                    text = "Downloading $selectedModel...",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}

@Composable
fun StatusIndicator(modelState: ModelState) {
    val color = when (modelState) {
        ModelState.LOADED -> MaterialTheme.colorScheme.primary
        ModelState.UNLOADED -> MaterialTheme.colorScheme.error
        else -> MaterialTheme.colorScheme.tertiary
    }

    val infiniteTransition = rememberInfiniteTransition(label = "status indicator")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.4f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "pulse animation",
    )

    val indicatorAlpha = if (modelState == ModelState.LOADED || modelState == ModelState.UNLOADED) 1f else alpha

    Surface(
        modifier = Modifier
            .size(10.dp)
            .alpha(indicatorAlpha),
        shape = CircleShape,
        color = color,
    ) {}
}
