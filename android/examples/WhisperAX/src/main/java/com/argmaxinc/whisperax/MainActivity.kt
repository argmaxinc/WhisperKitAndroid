//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.os.Build
import android.os.Bundle
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

class MainActivity : ComponentActivity() {
    companion object {
        const val TAG = "com.argmaxinc.whisperax"
    }

    private lateinit var manageExternalStorageLauncher: ActivityResultLauncher<android.content.Intent>
    private lateinit var viewModel: WhisperViewModel

    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        window.statusBarColor = android.graphics.Color.TRANSPARENT
        window.navigationBarColor = android.graphics.Color.TRANSPARENT

        manageExternalStorageLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult(),
        ) {
            if (!android.os.Environment.isExternalStorageManager()) {
                android.widget.Toast.makeText(
                    this,
                    "Storage permission is required for full functionality",
                    android.widget.Toast.LENGTH_LONG,
                ).show()
            }
        }

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
