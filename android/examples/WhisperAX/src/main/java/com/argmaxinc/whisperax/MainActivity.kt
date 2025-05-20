//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.view.WindowCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import android.Manifest

class MainActivity : ComponentActivity() {
    companion object {
        const val TAG = "com.argmaxinc.whisperax"
    }

    private lateinit var viewModel: WhisperViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        window.statusBarColor = android.graphics.Color.TRANSPARENT
        window.navigationBarColor = android.graphics.Color.TRANSPARENT

        viewModel = WhisperViewModel()
        viewModel.initContext(this)

        setContent {
            WhisperAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    MainScreen(viewModel = viewModel)
                }
            }
        }
    }
}

@Composable
fun MainScreen(viewModel: WhisperViewModel = viewModel()) {
    var currentScreen by remember { mutableStateOf("main") }

    when (currentScreen) {
        "main" -> {
            MainScreenContent(
                viewModel = viewModel,
                onNavigate = { screen ->
                    currentScreen = screen.lowercase()
                },
            )
        }

        "transcribe" -> {
            TranscribeScreen(
                viewModel = viewModel,
                onBackPressed = {
                    if (viewModel.isTranscribing.value) {
                        viewModel.stopTranscription()
                    }
                    currentScreen = "main"
                },
            )
        }
    }
}

@Composable
fun MainScreenContent(
    viewModel: WhisperViewModel,
    onNavigate: (String) -> Unit,
) {
    MainScreenContentImpl(viewModel, onNavigate)
}
