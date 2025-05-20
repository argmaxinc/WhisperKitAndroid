//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.Manifest
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Fullscreen
import androidx.compose.material.icons.filled.FullscreenExit
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch

@Composable
fun WaveformVisualization(
    energyValues: List<Float>,
    modifier: Modifier = Modifier,
) {
    val strokeWidth = remember { 2f }

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(100.dp)
            .padding(vertical = 8.dp),
    ) {
        val width = size.width
        val height = size.height
        val centerY = height / 2

        for (i in 0..4) {
            val y = height * i / 4
            drawLine(
                color = Color.Red.copy(alpha = 0.2f),
                start = Offset(0f, y),
                end = Offset(width, y),
                strokeWidth = 1f,
            )
        }

        if (energyValues.isNotEmpty()) {
            val path = Path()
            val startIndex = maxOf(0, energyValues.size - 300)
            val values = energyValues.subList(startIndex, energyValues.size)

            val xSpacing = width / (values.size - 1)

            path.moveTo(0f, centerY)

            values.forEachIndexed { index, energy ->
                val x = index * xSpacing
                val y = centerY - (energy * height / 2)
                path.lineTo(x, y)
            }

            drawPath(
                path = path,
                color = Color.Red,
                style = Stroke(width = strokeWidth),
            )

            val mirroredPath = Path()
            mirroredPath.moveTo(0f, centerY)

            values.forEachIndexed { index, energy ->
                val x = index * xSpacing
                val y = centerY + (energy * height / 2)
                mirroredPath.lineTo(x, y)
            }

            drawPath(
                path = mirroredPath,
                color = Color.Red,
                style = Stroke(width = strokeWidth),
            )
        }
    }
}

@SuppressLint("DefaultLocale")
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscribeScreen(
    viewModel: WhisperViewModel,
    onBackPressed: (() -> Unit)? = null,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val isTranscribing by viewModel.isTranscribing.collectAsState()
    val isRecording by viewModel.isRecording.collectAsState()
    val isInitializing by viewModel.isInitializing.collectAsState()
    val modelState by viewModel.modelState.collectAsState()
    val fullTranscript by viewModel.currentText.collectAsState()
    val effectiveRealTimeFactor by viewModel.effectiveRealTimeFactor.collectAsState()
    val effectiveSpeedFactor by viewModel.effectiveSpeedFactor.collectAsState()
    val tokensPerSecond by viewModel.tokensPerSecond.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    var isFullscreen by remember { mutableStateOf(false) }
    var showErrorDialog by remember { mutableStateOf(false) }

    LaunchedEffect(errorMessage) {
        showErrorDialog = errorMessage != null
    }

    val scrollState = rememberScrollState()

    LaunchedEffect(fullTranscript) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { isGranted ->
        if (isGranted) {
            scope.launch { viewModel.toggleRecording(false) }
        } else {
            Toast.makeText(context, "Microphone permission is required for recording", Toast.LENGTH_SHORT).show()
        }
    }

    fun checkAndRequestPermission() {
        when {
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> {
                scope.launch { viewModel.toggleRecording(false) }
            }
            else -> {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    val audioFilePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri: Uri? ->
        uri?.let { viewModel.transcribeFile(context, it) }
    }

    val readStoragePermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            audioFilePicker.launch("audio/*")
        } else {
            Toast.makeText(context, "Storage read permission is required to select audio files", Toast.LENGTH_SHORT).show()
        }
    }

    fun checkAndRequestReadPermission() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED -> {
                audioFilePicker.launch("audio/*")
            }
            else -> {
                readStoragePermissionLauncher.launch(permission)
            }
        }
    }

    if (showErrorDialog && errorMessage != null) {
        AlertDialog(
            onDismissRequest = { showErrorDialog = false },
            title = { Text("Error") },
            text = { Text(errorMessage ?: "") },
            confirmButton = {
                TextButton(onClick = { showErrorDialog = false }) {
                    Text("OK")
                }
            },
            containerColor = MaterialTheme.colorScheme.errorContainer,
            titleContentColor = MaterialTheme.colorScheme.onErrorContainer,
            textContentColor = MaterialTheme.colorScheme.onErrorContainer,
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(16.dp),
    ) {
        TopAppBar(
            title = { Text("Transcribe") },
            navigationIcon = {
                IconButton(onClick = { onBackPressed?.invoke() }) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                }
            },
            actions = {
                IconButton(
                    onClick = {
                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        val clip = ClipData.newPlainText("Transcription", fullTranscript)
                        clipboard.setPrimaryClip(clip)
                    },
                ) {
                    Icon(Icons.Default.ContentCopy, contentDescription = "Copy Text")
                }

                IconButton(onClick = { isFullscreen = !isFullscreen }) {
                    Icon(
                        if (isFullscreen) Icons.Default.FullscreenExit else Icons.Default.Fullscreen,
                        contentDescription = if (isFullscreen) "Exit Fullscreen" else "Enter Fullscreen",
                    )
                }
            },
        )
        Spacer(modifier = Modifier.height(8.dp))
        if (isRecording) {
            WaveformVisualization(
                energyValues = viewModel.bufferEnergy,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Card(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.1f),
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(16.dp),
            ) {
                Log.d("TranscribeDebug", "Rendering transcription box, fullTranscript length: ${fullTranscript.length}")
                if (isInitializing) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(48.dp),
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                } else if (fullTranscript.isNotBlank()) {
                    Log.d("TranscribeDebug", "Displaying transcript text")
                    Text(
                        text = fullTranscript,
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                } else if (errorMessage != null) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        Icon(
                            Icons.Default.Error,
                            contentDescription = "Error",
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(48.dp),
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Could not process the audio file.",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            OutlinedButton(
                                onClick = { showErrorDialog = true },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error,
                                ),
                            ) {
                                Text("Show Details")
                            }

                            Button(
                                onClick = { viewModel.resetState() },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary,
                                ),
                            ) {
                                Text("Try Again")
                            }
                        }
                    }
                } else {
                    Log.d("TranscribeDebug", "Displaying waiting message")
                    Text(
                        text = "Waiting for transcription...",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.align(Alignment.Center),
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (!isFullscreen) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.1f),
                ),
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        IconButton(
                            onClick = { viewModel.resetState() },
                            modifier = Modifier.size(32.dp),
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = "Reset", tint = MaterialTheme.colorScheme.primary)
                        }

                        Text(
                            text = "${String.format("%.3f", effectiveRealTimeFactor)} RTF",
                            style = MaterialTheme.typography.bodySmall,
                        )

                        Text(
                            text = "${String.format("%.1f", effectiveSpeedFactor)} Speed Factor",
                            style = MaterialTheme.typography.bodySmall,
                        )

                        Text(
                            text = "${tokensPerSecond.toInt()} tok/s",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Column(
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Button(
                        onClick = {
                            if (isTranscribing) {
                                viewModel.stopTranscription()
                            } else if (modelState == ModelState.LOADED) {
                                checkAndRequestReadPermission()
                            }
                        },
                        enabled = (modelState == ModelState.LOADED && !isRecording && errorMessage == null && !isInitializing) || (isTranscribing && !isInitializing),
                        modifier = Modifier
                            .size(72.dp)
                            .clip(MaterialTheme.shapes.medium),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isTranscribing) {
                                MaterialTheme.colorScheme.error
                            } else {
                                MaterialTheme.colorScheme.primaryContainer
                            },
                            contentColor = if (isTranscribing) {
                                MaterialTheme.colorScheme.onError
                            } else {
                                MaterialTheme.colorScheme.onPrimaryContainer
                            },
                        ),
                    ) {
                        Icon(
                            if (isTranscribing) Icons.Default.Stop else Icons.Default.AudioFile,
                            contentDescription = if (isTranscribing) "Stop Transcription" else "From File",
                            modifier = Modifier.size(64.dp),
                        )
                    }

                    Spacer(modifier = Modifier.width(16.dp))

                    Button(
                        onClick = {
                            if (isRecording) {
                                scope.launch { viewModel.toggleRecording(false) }
                            } else if (modelState == ModelState.LOADED) {
                                checkAndRequestPermission()
                            }
                        },
                        enabled = (modelState == ModelState.LOADED && !isTranscribing && errorMessage == null && !isInitializing) || (isRecording && !isInitializing),
                        modifier = Modifier
                            .size(72.dp)
                            .clip(CircleShape),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) {
                                MaterialTheme.colorScheme.error
                            } else {
                                MaterialTheme.colorScheme.primaryContainer
                            },
                            contentColor = if (isRecording) {
                                MaterialTheme.colorScheme.onError
                            } else {
                                MaterialTheme.colorScheme.onPrimaryContainer
                            },
                        ),
                    ) {
                        Icon(
                            if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                            contentDescription = if (isRecording) "Stop Recording" else "Start Recording",
                            modifier = Modifier.size(64.dp),
                        )
                    }
                }
            }
        }
    }
}
