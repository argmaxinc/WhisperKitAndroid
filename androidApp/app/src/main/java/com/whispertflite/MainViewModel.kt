package com.whispertflite

import android.content.Context
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whispertflite.asr.Player
import com.whispertflite.asr.Recorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.system.measureTimeMillis


class MainViewModel : ViewModel() {

    lateinit var statusState: MutableState<String>
    lateinit var resultState: MutableState<String>
    lateinit var modelDestFolder: File

    // Constant strings
    private val defaultAudioFileName = "jfk.wav"
    private val whisperFolderName = "openai_whisper-tiny"
    private val microphoneInputFileName = "MicInput.wav"

    private val _waveFileNamesState = mutableStateOf<List<String>>(emptyList())
    val waveFileNamesState: State<List<String>> get() = _waveFileNamesState

    var mPlayer: Player? = null
    var waveFile: File? = null
    private var mRecorder: Recorder? = null

    var sdcardDataFolder: File? = null
    val selectedWavFilename = mutableStateOf(defaultAudioFileName)

    private lateinit var whisperKit: WhisperKitNative

    fun initialize(context: Context) {
        val cacheDir = context.cacheDir
        Log.d("Cache", cacheDir.absolutePath)
        val nativeLibsDir = context.applicationInfo.nativeLibraryDir
        copyDataToSdCardFolder(context)
        loadAudioFileNames(sdcardDataFolder!!.absolutePath)

        val audioPath = sdcardDataFolder!!.absolutePath + "/" + selectedWavFilename.value

        mRecorder = Recorder(context)
        mPlayer = Player(context)

        resultState = mutableStateOf("")
        statusState = mutableStateOf("LOADING MODEL (This may take a few minutes when using QNN)")

        waveFile = File(sdcardDataFolder!!.absolutePath + "/" + microphoneInputFileName)

        viewModelScope.launch(Dispatchers.IO) {
            whisperKit = WhisperKitNative(modelDestFolder.absolutePath, audioPath, ".", nativeLibsDir,  4)
            statusState.value = "IDLE"
        }
    }

    private fun copyDataToSdCardFolder(context: Context) {
        sdcardDataFolder = context.getExternalFilesDir(null)
        modelDestFolder = File(sdcardDataFolder!!.absolutePath + "/" + whisperFolderName)
        copyAssetsToSdcard(context, sdcardDataFolder!!, arrayOf( "wav", "m4a"))
        copyAssetsDirectory(context, whisperFolderName, modelDestFolder)
        Log.d("ASDF", "Assets directory copied to emulated storage")
    }


    fun runInference() {
        statusState.value = "TRANSCRIBING"
        viewModelScope.launch(Dispatchers.IO) {
            lateinit var transcriptOutput: String
            val time = measureTimeMillis {
                transcriptOutput = whisperKit.transcribe(sdcardDataFolder!!.absolutePath + "/" + selectedWavFilename.value)
            }

            statusState.value = "IDLE  |  Last transcript took $time ms"
            Log.d("Transcript", transcriptOutput)
            resultState.value = transcriptOutput
        }
    }

    fun releaseModel() {
        whisperKit.release()
    }

    fun startRecording() {
        mRecorder?.setFilePath(waveFile?.absolutePath)
        mRecorder?.setFolderPath(sdcardDataFolder?.absolutePath)
        mRecorder?.start()
    }

    fun stopRecording() {
        mRecorder?.stop()
    }

    private fun copyAssetsToSdcard(context: Context, destFolder: File, extensions: Array<String>) {
        val assetManager = context.assets

        try {
            val assetFiles = assetManager.list("") ?: return
            if (!destFolder.exists()) {
                destFolder.mkdirs()
            }
            for (assetFileName in assetFiles) {
                if (extensions.any { assetFileName.endsWith(".$it") }) {
                    val outFile = File(destFolder, assetFileName)
                    if (outFile.exists()) continue

                    assetManager.open(assetFileName).use { inputStream ->
                        FileOutputStream(outFile).use { outputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    private fun copyAssetsDirectory(
        context: Context,
        assetDir: String,
        destFolder: File
    ) {
        val assetManager = context.assets
        Log.d("ASDF", destFolder.absolutePath)
        try {
            val assetFiles = assetManager.list(assetDir) ?: return

            if (!destFolder.exists()) {
                destFolder.mkdirs()
            }

            for (assetFileName in assetFiles) {
                Log.d("ASDF", assetFileName)
                val assetPath = if (assetDir.isEmpty()) assetFileName else "$assetDir/$assetFileName"
                val outFile = File(destFolder, assetFileName)
                if (outFile.exists()) continue

                assetManager.open(assetPath).use { inputStream ->
                    FileOutputStream(outFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun loadAudioFileNames(folderPath: String) {
        val folder = File(folderPath)
        if (folder.exists() && folder.isDirectory) {
            // Retrieve only file names (excluding directories)
            val fileNames = folder.listFiles()
                ?.filter { it.isFile }
                ?.map { it.name }
                ?: emptyList()

            // Update the state with the list of file names
            _waveFileNamesState.value = fileNames
        } else {
            // Folder doesn't exist or is not a directory
            _waveFileNamesState.value = emptyList()
        }
    }
}