/*
 * Copyright 2023 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.whispertflite

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

/** Sample activity to test the TFLite C API. */
class MainActivity : ComponentActivity() {

    object LibraryLoader {
        private val libraries = arrayOf(
            "avutil",       // Base library for multimedia utilities
            "swresample",   // For audio resampling (depends on avutil)
            "avcodec",      // For multimedia decoding (depends on avutil)
            "avformat",     // For multimedia format handling (depends on avcodec)

            // QNN libraries
            "QnnSystem",    // Core system library for QNN
            "QnnGpu",       // GPU support for QNN
            "QnnDsp",       // DSP support for QNN
            "QnnDspV66Skel",
            "QnnDspV66Stub",
            "QnnHtp",
            "QnnHtpPrepare",
            "QnnHtpV68Skel",
            "QnnHtpV68Stub",
            "QnnHtpV69Skel",
            "QnnHtpV69Stub",
            "QnnHtpV73Skel",
            "QnnHtpV73Stub",
            "QnnHtpV75Skel",
            "QnnHtpV75Stub",

            // TensorFlow Lite delegate for QNN
            "QnnTFLiteDelegate",
            "tensorflowlite_gpu_delegate", // GPU support for TensorFlow Lite

            // Native WhisperKit library
            "whisperkit",
            "native-whisper"
        )

        fun loadAllLibraries() {
            libraries.forEach { lib ->
                try {
                    System.loadLibrary(lib)
                    println("Loaded library: lib$lib.so")
                } catch (e: UnsatisfiedLinkError) {
                    println("Failed to load library: lib$lib.so")
                    e.printStackTrace()
                }
            }
        }
    }
    private lateinit var mainViewModel: MainViewModel

    private val TAG = "MainActivity"
    private var isRecording = false

    companion object {
        init {

            LibraryLoader.loadAllLibraries()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestRecordPermission()
        actionBar?.hide()
        mainViewModel = ViewModelProvider(this)[MainViewModel::class.java]
        mainViewModel.initialize(this)
        setContent {
            MainScreen(this)
        }
        checkRecordPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        mainViewModel.releaseModel()
    }

    @Composable
    fun MainScreen(context: Context) {

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Gray)
                .padding(16.dp)

        ) {
            InputFileSelectionSection()
            ActionButtonSection(
                transcribeAction = {mainViewModel.runInference()},
                recordAction = {recordAction()},
                playAction = {mainViewModel.mPlayer?.initializePlayer(mainViewModel.sdcardDataFolder?.absolutePath + "/" + mainViewModel.selectedWavFilename.value)}
            )
            StatusSection()
            Divider(modifier = Modifier.padding(vertical = 16.dp))
            ResultsSection()
            Divider(modifier = Modifier.padding(vertical = 16.dp))
            LogcatScreen()
        }
    }

    @Composable
    fun InputFileSelectionSection() {
        val wavFileList by mainViewModel.waveFileNamesState
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            color = Color(0xFFC4C4C4),
            shape = RoundedCornerShape(8.dp),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp), // Content padding
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "SELECT AUDIO FILE",
                    modifier = Modifier
                        .width(130.dp),
                    color = Color(0xFFFFFFFF),
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontWeight = FontWeight.Bold
                    )
                )
                DropdownMenuComponent(
                    items = wavFileList,
                    modifier = Modifier.weight(5f),
                    selectedItemState = mainViewModel.selectedWavFilename
                )
            }
        }
    }

    @Composable
    fun ActionButtonSection(
        transcribeAction: () -> Unit,
        recordAction: () -> Unit,
        playAction: () -> Unit,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { recordAction() },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFB2B2B2),
                        contentColor = Color.White,
                        disabledContainerColor = Color.Gray,
                        disabledContentColor = Color.White
                )
                ) {
                    Text(text = if (!isRecording) "Record" else "Stop") // TODO: FIX
                }
                Button(
                    onClick = { playAction() },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFB2B2B2),
                        contentColor = Color.White,
                        disabledContainerColor = Color.Gray,
                        disabledContentColor = Color.White
                    )
                ) {
                    Text(text = "Play")
                }
            }

            Button(
                onClick = { transcribeAction() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFFF9800), // Orange background
                    contentColor = Color.White,         // White text
                    disabledContainerColor = Color.Gray,
                    disabledContentColor = Color.White
                )
            ) {
                Text(text = "Transcribe")
            }
        }
    }

    @Composable
    fun StatusSection() {
        val status by mainViewModel.statusState
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Status: ",
                modifier = Modifier.padding(end = 8.dp)
            )
            Text(
                text = status,
                modifier = Modifier.weight(1f)
            )
        }
    }

    @Composable
    fun ResultsSection() {
        val result by mainViewModel.resultState
        Column(
            modifier = Modifier
                .fillMaxHeight(0.5f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())

        ) {
            Text(
                text = result,
                modifier = Modifier.fillMaxWidth(),
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }

    @Composable
    fun DropdownMenuComponent(
        items: List<String>,
        modifier: Modifier = Modifier,
        selectedItemState: MutableState<String>
        ) {
        var expanded by remember { mutableStateOf(false) }


        Box(modifier = modifier) {
            TextButton(onClick = { expanded = true }) {
                Text(selectedItemState.value,
                    color = Color(0xFFFFBA44),
                    style = MaterialTheme.typography.bodyLarge.copy(
                        fontWeight = FontWeight.Bold
                    )
                    )
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                items.forEach { item ->
                    DropdownMenuItem(
                        text = { Text(item) },
                        onClick = {
                            selectedItemState.value = item
                            expanded = false
                        }
                    )
                }
            }
        }
    }

    private fun checkRecordPermission() {
        val permission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        if (permission == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Record permission is granted")
        } else {
            Log.d(TAG, "Requesting record permission")
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 0)
        }
    }


    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            // Handle the permissions result
            permissions.entries.forEach { entry ->
                when (entry.key) {
                    Manifest.permission.RECORD_AUDIO -> {
                        if (entry.value) {
                            Log.d(TAG, "Record permission is granted")
                        } else {
                            Log.d(TAG, "Record permission is not granted")
                        }
                    }
                    Manifest.permission.WRITE_EXTERNAL_STORAGE -> {
                        if (entry.value) {
                            Log.d(TAG, "Write permission is granted")
                        } else {
                            Log.d(TAG, "Write permission is not granted")
                        }
                    }
                }
            }
        }

    private fun requestRecordPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)) {
            // Optionally show rationale to the user explaining why the permission is needed
            Log.d(TAG, "Displaying permission rationale")
        } else {
            // Directly request the permission
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
        }
    }

    private fun recordAction() {
        if(!isRecording) {
            mainViewModel.startRecording()
            Log.d(TAG, "Started recording")
            isRecording = true
        } else {
            mainViewModel.stopRecording()
            Log.d(TAG, "Stopped recording")
            isRecording = false
        }
    }

    @Composable
    fun LogcatScreen() {
        // State list to hold log lines
        val logs = remember { mutableStateListOf<String>() }

        // Launch a coroutine to read from logcat
        LaunchedEffect(Unit) {
            withContext(Dispatchers.IO) {
                try {
                    // Execute the logcat command
                    val process = Runtime.getRuntime().exec("logcat tflite:V *:S")
                    val reader = BufferedReader(InputStreamReader(process.inputStream))
                    // Continuously read new lines
                    while (isActive) {
                        val line = reader.readLine() ?: break
                        // Post the line to the UI thread
                        withContext(Dispatchers.Main) {
                            logs.add(line)
                            // Optionally, limit the size of the logs list
                            if (logs.size > 1000) { // for example, keep only the latest 1000 lines
                                logs.removeAt(0)
                            }
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }

        // Display the logs in a scrollable list
        LazyColumn(
            modifier = Modifier.fillMaxWidth(),
            contentPadding = PaddingValues(16.dp)
        ) {
            items(logs) { log ->
                Text(text = log)
            }
        }
    }

}