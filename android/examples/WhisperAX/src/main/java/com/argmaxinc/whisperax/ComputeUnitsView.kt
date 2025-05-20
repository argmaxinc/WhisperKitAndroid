//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.unit.dp
import com.argmaxinc.whisperkit.ExperimentalWhisperKit
import com.argmaxinc.whisperkit.WhisperKit

@OptIn(ExperimentalWhisperKit::class)
enum class ComputeUnits(val displayName: String, val backendValue: Int) {
    CPU_ONLY("CPU", WhisperKit.Builder.CPU_ONLY),
    CPU_AND_GPU("GPU", WhisperKit.Builder.CPU_AND_GPU),
    CPU_AND_NPU("NPU", WhisperKit.Builder.CPU_AND_NPU),
}

@Composable
fun ComputeUnitsView(viewModel: WhisperViewModel) {
    val modelState by viewModel.modelState.collectAsState()
    val encoderState by viewModel.encoderState.collectAsState()
    val decoderState by viewModel.decoderState.collectAsState()
    val isEnabled = modelState == ModelState.LOADED || modelState == ModelState.UNLOADED

    var whisperKitExpanded by remember { mutableStateOf(true) }
    var speakerKitExpanded by remember { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxWidth()) {
        ComputeUnitDisclosureGroup(
            title = "Compute Units",
            isExpanded = whisperKitExpanded,
            onExpandedChange = {
                whisperKitExpanded = it
            },
            enabled = isEnabled,
        ) {
            Column(modifier = Modifier.padding(start = 8.dp, top = 8.dp)) {
                ComputeUnitRow(
                    title = "Audio Encoder",
                    currentState = encoderState,
                    currentUnit = viewModel.encoderComputeUnits.collectAsState().value,
                    onUnitSelected = { viewModel.setEncoderComputeUnits(it) },
                    enabled = isEnabled,
                )

                Spacer(modifier = Modifier.height(8.dp))

                ComputeUnitRow(
                    title = "Text Decoder",
                    currentState = decoderState,
                    currentUnit = viewModel.decoderComputeUnits.collectAsState().value,
                    onUnitSelected = { viewModel.setDecoderComputeUnits(it) },
                    enabled = isEnabled,
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        ComputeUnitDisclosureGroup(
            title = "SpeakerKit",
            isExpanded = speakerKitExpanded,
            onExpandedChange = {
                speakerKitExpanded = it
                if (it) whisperKitExpanded = false
            },
            enabled = isEnabled,
        ) {
            Column(modifier = Modifier.padding(start = 8.dp, top = 8.dp)) {
                ComputeUnitRow(
                    title = "Segmenter",
                    currentState = encoderState,
                    currentUnit = viewModel.segmenterComputeUnits.collectAsState().value,
                    onUnitSelected = { viewModel.setSegmenterComputeUnits(it) },
                    enabled = false,
                )

                Spacer(modifier = Modifier.height(8.dp))

                ComputeUnitRow(
                    title = "Embedder",
                    currentState = decoderState,
                    currentUnit = viewModel.embedderComputeUnits.collectAsState().value,
                    onUnitSelected = { viewModel.setEmbedderComputeUnits(it) },
                    enabled = false,
                )
            }
        }
    }
}

@Composable
fun ComputeUnitDisclosureGroup(
    title: String,
    isExpanded: Boolean,
    onExpandedChange: (Boolean) -> Unit,
    enabled: Boolean = true,
    content: @Composable () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    color = if (enabled) {
                        MaterialTheme.colorScheme.onSurface
                    } else {
                        MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    },
                )

                Spacer(modifier = Modifier.weight(1f))

                IconButton(
                    onClick = { onExpandedChange(!isExpanded) },
                    enabled = enabled,
                ) {
                    Icon(
                        imageVector = Icons.Default.KeyboardArrowDown,
                        contentDescription = if (isExpanded) "Collapse" else "Expand",
                        modifier = Modifier.rotate(if (isExpanded) 180f else 0f),
                        tint = if (enabled) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        },
                    )
                }
            }

            AnimatedVisibility(visible = isExpanded) {
                content()
            }
        }
    }
}

@Composable
fun ComputeUnitRow(
    title: String,
    currentState: ModelState,
    currentUnit: ComputeUnits,
    onUnitSelected: (ComputeUnits) -> Unit,
    enabled: Boolean = true,
) {
    val infiniteTransition = rememberInfiniteTransition(label = "loading animation")
    val colorAlpha = infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.3f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1000),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "loading alpha",
    )

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .alpha(if (currentState == ModelState.LOADING) colorAlpha.value else 1f),
        ) {
            val color = when (currentState) {
                ModelState.LOADED -> MaterialTheme.colorScheme.primary
                ModelState.UNLOADED -> MaterialTheme.colorScheme.error
                else -> MaterialTheme.colorScheme.tertiary
            }

            Surface(
                modifier = Modifier.size(10.dp),
                color = color,
                shape = CircleShape,
            ) {}
        }

        Spacer(modifier = Modifier.width(8.dp))

        Text(
            text = title,
            style = MaterialTheme.typography.bodyMedium,
        )

        Spacer(modifier = Modifier.weight(1f))

        Box {
            var expanded by remember { mutableStateOf(false) }

            Button(
                onClick = { if (enabled) expanded = true },
                enabled = enabled,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
                modifier = Modifier.width(150.dp),
            ) {
                Text(currentUnit.displayName)
            }

            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
            ) {
                ComputeUnits.values().forEach { unit ->
                    DropdownMenuItem(
                        text = { Text(unit.displayName) },
                        onClick = {
                            onUnitSelected(unit)
                            expanded = false
                        },
                    )
                }
            }
        }
    }
}
