//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

import android.content.ContentResolver
import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.argmaxinc.whisperkit.ExperimentalWhisperKit
import com.argmaxinc.whisperkit.WhisperKit
import com.argmaxinc.whisperkit.WhisperKit.TextOutputCallback
import com.argmaxinc.whisperkit.WhisperKitException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class TranscriptionSegment(
    val text: String,
    val start: Float,
    val end: Float,
    val tokens: List<Int> = emptyList(),
)

data class TranscriptionResult(
    val text: String = "",
    val segments: List<TranscriptionSegment> = emptyList(),
)

@OptIn(ExperimentalWhisperKit::class)
class WhisperViewModel : ViewModel() {
    companion object {
        const val TAG = "WhisperViewModel"
    }

    private lateinit var appContext: Context

    private val callbacks = DecoderCallbacks()
    private lateinit var audioDecoder: PCMDecoder
    private var totalPCMdata: Int = 0
    private var latestTime: Int = 0
    private var samplerate: Int = 0
    private var channels: Int = 0
    private var duration: Long = 0
    private var selectedFileUri: Uri? = null
    private var startTime: Long = 0
    private var firstTokenReceived = false
    private var firstTokenTimestamp: Long = 0
    private var totalTokens = 0
    private var lastTokenTimestamp: Long = 0
    private val allText = StringBuilder()
    private var nativeLibraryDir: String = ""
    private var cacheDir: String = ""
    private val MIN_FRAME_BYTES = 960_000

    private val _modelState = MutableStateFlow(ModelState.UNLOADED)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _encoderState = MutableStateFlow(ModelState.UNLOADED)
    val encoderState: StateFlow<ModelState> = _encoderState.asStateFlow()

    private val _decoderState = MutableStateFlow(ModelState.UNLOADED)
    val decoderState: StateFlow<ModelState> = _decoderState.asStateFlow()

    private val _loadingProgress = MutableStateFlow(0f)
    val loadingProgress: StateFlow<Float> = _loadingProgress.asStateFlow()

    private val _selectedModel = MutableStateFlow("")
    val selectedModel: StateFlow<String> = _selectedModel.asStateFlow()

    private val _isModelMultilingual = MutableStateFlow(true)

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private var isRecordingActionInProgress = false
    private var lastRecordingActionTime = 0L
    private val DEBOUNCE_TIMEOUT = 1000L // 1 second debounce timeout

    private val _isTranscribing = MutableStateFlow(false)
    val isTranscribing: StateFlow<Boolean> = _isTranscribing.asStateFlow()

    private val _isInitializing = MutableStateFlow(false)
    val isInitializing: StateFlow<Boolean> = _isInitializing.asStateFlow()

    private val _bufferSeconds = MutableStateFlow(0.0)

    private val _bufferEnergy = mutableStateListOf<Float>()
    val bufferEnergy: List<Float> = _bufferEnergy

    private val _currentText = MutableStateFlow("")
    val currentText: StateFlow<String> = _currentText.asStateFlow()

    private val _effectiveRealTimeFactor = MutableStateFlow(0.0)
    val effectiveRealTimeFactor: StateFlow<Double> = _effectiveRealTimeFactor.asStateFlow()

    private val _effectiveSpeedFactor = MutableStateFlow(0.0)
    val effectiveSpeedFactor: StateFlow<Double> = _effectiveSpeedFactor.asStateFlow()

    private val _tokensPerSecond = MutableStateFlow(0.0)
    val tokensPerSecond: StateFlow<Double> = _tokensPerSecond.asStateFlow()

    private val _selectedTask = MutableStateFlow("transcribe")
    val selectedTask: StateFlow<String> = _selectedTask.asStateFlow()

    private val _selectedLanguage = MutableStateFlow("english")
    val selectedLanguage: StateFlow<String> = _selectedLanguage.asStateFlow()

    private val _selectedTab = MutableStateFlow("Transcribe")
    val selectedTab: StateFlow<String> = _selectedTab.asStateFlow()

    val availableModels = mutableStateListOf<String>()

    val localModels = mutableStateListOf("tiny", "base", "small")

    private val _downloadedModels = mutableStateListOf<String>()
    val downloadedModels: List<String> = _downloadedModels

    private val _specializedModels = mutableStateListOf<String>()

    val availableLanguages = listOf(
        "english", "chinese", "german", "spanish", "russian", "korean",
        "french", "japanese", "portuguese", "turkish", "polish", "catalan",
    )

    private val _firstTokenTime = MutableStateFlow(0.0)
    val firstTokenTime: StateFlow<Double> = _firstTokenTime.asStateFlow()

    private val _pipelineStart = MutableStateFlow(0.0)
    val pipelineStart: StateFlow<Double> = _pipelineStart.asStateFlow()

    private val _audioSampleDuration = MutableStateFlow(0.0)
    val audioSampleDuration: StateFlow<Double> = _audioSampleDuration.asStateFlow()

    private val _transcriptionDuration = MutableStateFlow(0.0)
    val transcriptionDuration: StateFlow<Double> = _transcriptionDuration.asStateFlow()

    private val _totalProcessTime = MutableStateFlow(0.0)
    val totalProcessTime: StateFlow<Double> = _totalProcessTime.asStateFlow()

    private val _enableTimestamps = MutableStateFlow(true)
    val enableTimestamps: StateFlow<Boolean> = _enableTimestamps.asStateFlow()

    private val _encoderComputeUnits = MutableStateFlow(ComputeUnits.CPU_AND_NPU)
    val encoderComputeUnits: StateFlow<ComputeUnits> = _encoderComputeUnits.asStateFlow()

    private val _decoderComputeUnits = MutableStateFlow(ComputeUnits.CPU_AND_NPU)
    val decoderComputeUnits: StateFlow<ComputeUnits> = _decoderComputeUnits.asStateFlow()

    private val _segmenterComputeUnits = MutableStateFlow(ComputeUnits.CPU_AND_GPU)
    val segmenterComputeUnits: StateFlow<ComputeUnits> = _segmenterComputeUnits.asStateFlow()

    private val _embedderComputeUnits = MutableStateFlow(ComputeUnits.CPU_AND_GPU)
    val embedderComputeUnits: StateFlow<ComputeUnits> = _embedderComputeUnits.asStateFlow()

    private var audioRecorder: AudioRecorder? = null
    private var energyCollectJob: Job? = null

    private lateinit var whisperKit: WhisperKit

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    init {
        resetState()
        listModels()
    }

    fun initContext(context: Context) {
        appContext = context.applicationContext
        nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        cacheDir = context.cacheDir.absolutePath
    }

    fun onTextOutput(what: Int, timestamp: Float, msg: String) {
        when (what) {
            TextOutputCallback.MSG_INIT -> {
                Log.i(MainActivity.TAG, "TFLite initialized: $msg")
                startTime = System.currentTimeMillis()
                _pipelineStart.value = startTime.toDouble() / 1000.0
                _isInitializing.value = false
            }

            TextOutputCallback.MSG_TEXT_OUT -> {
                Log.i(MainActivity.TAG, "TEXT OUT THREAD")
                if (msg.isNotEmpty()) {
                    if (!firstTokenReceived) {
                        firstTokenReceived = true
                        firstTokenTimestamp = System.currentTimeMillis()
                        _firstTokenTime.value = (firstTokenTimestamp - startTime).toDouble() / 1000.0
                    }

                    val newTokens = msg.length / 4
                    totalTokens += newTokens

                    val currentTime = System.currentTimeMillis()
                    val elapsedSeconds = (currentTime - startTime).toDouble() / 1000.0
                    if (elapsedSeconds > 0) {
                        _tokensPerSecond.value = totalTokens / elapsedSeconds

                        updateRealtimeMetrics(elapsedSeconds)
                    }

                    lastTokenTimestamp = currentTime
                    updateTranscript(msg)
                }
            }

            TextOutputCallback.MSG_CLOSE -> {
                Log.i(MainActivity.TAG, "Transcription completed.")
                if (msg.isNotEmpty()) {
                    val newTokens = msg.length / 4
                    totalTokens += newTokens

                    val totalTime = (System.currentTimeMillis() - startTime).toDouble() / 1000.0
                    if (totalTime > 0) {
                        _tokensPerSecond.value = totalTokens / totalTime

                        updateRealtimeMetrics(totalTime)
                    }

                    updateTranscript(msg)
                }
            }

            else -> {
                Log.w(MainActivity.TAG, "Unknown message code: $what")
            }
        }
    }

    private fun updateTranscript(chunkText: String, withTimestamps: Boolean = false) {
        var processedText = chunkText

        val timestamps = if (withTimestamps) {
            val timestampPattern = "<\\|(\\d+\\.\\d+)\\|>".toRegex()
            val timestampMatches = timestampPattern.findAll(chunkText).toList()
            timestampMatches.map { it.groupValues[1].toFloat() }
        } else {
            emptyList()
        }

        if (!withTimestamps) {
            processedText = processedText
                .replace("<\\|[^>]*\\|>".toRegex(), "")
                .trim()
        } else {
            processedText = processedText.trim()
        }

        if (processedText.isNotEmpty()) {
            if (allText.isNotEmpty()) {
                allText.append("\n")
            }

            allText.append(processedText)
        }

        Log.d("TranscribeDebug", "Full text after append: $allText")

        viewModelScope.launch(Dispatchers.Main) {
            _currentText.value = allText.toString().trim()
        }
    }

    fun listModels() {
        viewModelScope.launch {
            val modelDirs = listOf(
                // TODO: enable when models are ready
                // WhisperKit.Builder.OPENAI_TINY_EN,
                // WhisperKit.Builder.OPENAI_BASE_EN,
                // WhisperKit.Builder.OPENAI_SMALL_EN,
                WhisperKit.Builder.QUALCOMM_TINY_EN,
                WhisperKit.Builder.QUALCOMM_BASE_EN,
                // WhisperKit.Builder.QUALCOMM_SMALL_EN
            )
            availableModels.clear()
            availableModels.addAll(modelDirs)

            if (_selectedModel.value.isEmpty()) {
                _selectedModel.update {
                    modelDirs.first()
                }
            }
        }
    }

    fun loadModel(model: String) {
        viewModelScope.launch {
            try {
                _modelState.value = ModelState.LOADING
                _loadingProgress.value = 0f
                whisperKit = WhisperKit.Builder().setModel(model).setApplicationContext(appContext)
                    .setCallback(::onTextOutput)
                    .setEncoderBackend(_encoderComputeUnits.value.backendValue)
                    .setDecoderBackend(_decoderComputeUnits.value.backendValue)
                    .build()
                whisperKit.loadModel().collect { progress ->
                    _loadingProgress.value = progress.fractionCompleted
                    if (!_downloadedModels.contains(model)) {
                        _modelState.value = ModelState.DOWNLOADING
                    }
                }

                if (!_specializedModels.contains(model)) {
                    withContext(Dispatchers.Main) {
                        Log.i(TAG, "STARTING SPECIALIZATION for model: $model")
                        _loadingProgress.value = 0.0f
                        _modelState.value = ModelState.PREWARMING
                        delay(300)
                    }

                    withContext(Dispatchers.IO) {
                        val specializationBytes = ByteArray(30 * AudioRecorder.SAMPLE_RATE * 2) { 0 }
                        whisperKit.init(
                            frequency = AudioRecorder.SAMPLE_RATE,
                            channels = 1,
                            duration = 0,
                        )
                        Log.i(TAG, "Running specialization transcribe...")
                        whisperKit.transcribe(specializationBytes)
                        Log.i(TAG, "Specialization transcribe completed")
                        whisperKit.deinitialize()
                        _specializedModels.add(model)
                    }

                    Log.i(TAG, "SPECIALIZATION COMPLETED for model: $model")
                }

                withContext(Dispatchers.Main) {
                    _loadingProgress.value = 1.0f
                    _modelState.value = ModelState.LOADED
                    _selectedModel.value = model
                    _encoderState.value = ModelState.LOADED
                    _decoderState.value = ModelState.LOADED
                    if (!_downloadedModels.contains(model)) {
                        _downloadedModels.add(model)
                    }
                }
            } catch (e: Exception) {
                _modelState.value = ModelState.UNLOADED
                Log.e(MainActivity.TAG, "Error loading model", e)
            }
        }
    }

    fun selectModel(model: String) {
        _selectedModel.value = model
        _modelState.value = ModelState.UNLOADED
        _encoderState.value = ModelState.UNLOADED
        _decoderState.value = ModelState.UNLOADED
    }

    fun resetState() {
        viewModelScope.launch {
            if (_isTranscribing.value) {
                stopTranscription()
            } else {
                stopRecording()
            }

            _isTranscribing.value = false
            _currentText.value = ""
            _bufferEnergy.clear()
            _bufferSeconds.value = 0.0
            _errorMessage.value = null
            allText.clear()

            firstTokenReceived = false
            firstTokenTimestamp = 0
            totalTokens = 0
            lastTokenTimestamp = 0
            _tokensPerSecond.value = 0.0
            _firstTokenTime.value = 0.0
            _pipelineStart.value = 0.0
            _effectiveRealTimeFactor.value = 0.0
            _effectiveSpeedFactor.value = 0.0
            _audioSampleDuration.value = 0.0
            _transcriptionDuration.value = 0.0
            _totalProcessTime.value = 0.0
        }
    }

    fun transcribeFile(context: Context, uri: Uri) {
        resetState()
        viewModelScope.launch {
            _errorMessage.value = null
            _isTranscribing.value = true

            try {
                val filePath = getRealPathFromUri(context, uri)
                if (filePath != null) {
                    selectedFileUri = uri
                    startTranscription(context, filePath)
                } else {
                    val errorMsg = "Could not access the selected file. Please try another file."
                    Log.e(MainActivity.TAG, "Could not get path from URI")
                    _errorMessage.value = errorMsg
                    _isTranscribing.value = false
                }
            } catch (e: Exception) {
                val errorMsg = "Error loading file: ${e.message ?: "Unknown error"}"
                Log.e(MainActivity.TAG, "Error during transcription", e)
                _errorMessage.value = errorMsg
                _isTranscribing.value = false
            }
        }
    }

    private suspend fun startTranscription(context: Context, filePath: String) {
        withContext(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) {
                    _currentText.value = ""
                    _isInitializing.value = true
                    _errorMessage.value = null
                }

                allText.clear()
                Log.d("TranscribeDebug", "Starting decode for file: $filePath")

                val file = File(filePath)
                if (!file.exists() || !file.canRead()) {
                    throw IOException("Cannot access file: $filePath")
                }

                try {
                    audioDecoder = PCMDecoder(callbacks)
                    audioDecoder.decodeFileStart(filePath)
                } catch (e: Exception) {
                    throw IOException("Failed to decode audio file. The file may be corrupted or in an unsupported format.", e)
                }

                try {
                    whisperKit.init(frequency = samplerate, channels = channels, duration = duration)
                } catch (e: Exception) {
                    throw RuntimeException("Failed to initialize transcription engine", e)
                }

                try {
                    audioDecoder.waitInLoop()
                    audioDecoder.decodeStopRelease()
                } catch (e: Exception) {
                    throw IOException("Error processing audio data", e)
                }

                displayPerformanceMetrics(filePath, startTime, System.currentTimeMillis())
            } catch (e: Exception) {
                Log.e(MainActivity.TAG, "Error during transcription", e)
                withContext(Dispatchers.Main) {
                    _isTranscribing.value = false
                    _isInitializing.value = false
                    _errorMessage.value = e.message ?: "Failed to process audio file"
                }
            }
        }
    }

    private suspend fun startTranscription(context: Context, pcmBytes: ByteArray) {
        var totalBytesSent = 0
        val startTime = System.currentTimeMillis()

        withContext(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) {
                    _currentText.value = ""
                    _isInitializing.value = true
                    _isTranscribing.value = true
                    _errorMessage.value = null
                }

                try {
                    whisperKit.init(
                        frequency = AudioRecorder.SAMPLE_RATE,
                        channels = 1,
                        duration = 0,
                    )
                } catch (e: WhisperKitException) {
                    Log.e(TAG, "Failed to init WhisperKit: ${e.message}")
                    withContext(Dispatchers.Main) {
                        _isInitializing.value = false
                        _isTranscribing.value = false
                        _errorMessage.value = "Failed to initialize transcription: ${e.message}"
                    }
                    return@withContext
                }

                withContext(Dispatchers.Main) {
                    _isInitializing.value = false
                }

                val chunkBytes = 3200
                var offset = 0
                while (offset < pcmBytes.size) {
                    val end = minOf(offset + chunkBytes, pcmBytes.size)
                    val slice = pcmBytes.copyOfRange(offset, end)
                    val bufferedSecs = whisperKit.transcribe(slice)
                    totalBytesSent += slice.size
                    offset = end
                }

                val remainder = totalBytesSent % MIN_FRAME_BYTES
                if (remainder != 0) {
                    val pad = ByteArray(MIN_FRAME_BYTES - remainder) { 0 }
                    whisperKit.transcribe(pad)
                    totalBytesSent += pad.size
                }

                val afterInitTime = System.currentTimeMillis()
                whisperKit.deinitialize()

                val audioDuration = (pcmBytes.size / (AudioRecorder.SAMPLE_RATE * 2)).toDouble()
                val inferenceTime = (System.currentTimeMillis() - afterInitTime).toDouble() / 1000.0
                val totalTime = (System.currentTimeMillis() - startTime).toDouble() / 1000.0

                withContext(Dispatchers.Main) {
                    _audioSampleDuration.value = audioDuration
                    _transcriptionDuration.value = inferenceTime
                    _totalProcessTime.value = totalTime
                    _effectiveRealTimeFactor.value = totalTime / audioDuration
                    _effectiveSpeedFactor.value = audioDuration / totalTime
                    _isTranscribing.value = false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during transcription of recorded audio", e)
                withContext(Dispatchers.Main) {
                    _isInitializing.value = false
                    _isTranscribing.value = false
                    _errorMessage.value = "Error processing recorded audio: ${e.message}"
                }
            }
        }
    }

    private fun displayPerformanceMetrics(filePath: String, initTime: Long, afterInitTime: Long) {
        viewModelScope.launch {
            try {
                val now = System.currentTimeMillis()
                val inferenceTime = (now - afterInitTime).toDouble() / 1000.0
                withContext(Dispatchers.Main) {
                    _transcriptionDuration.value = inferenceTime
                    _isTranscribing.value = false
                }
            } catch (e: Exception) {
                Log.e(MainActivity.TAG, "Error displaying performance metrics", e)
                _isTranscribing.value = false
            }
        }
    }

    private fun getAudioDuration(filePath: String): Double? {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(filePath)
            val duration =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
                    ?: 0L
            retriever.release()
            duration.toDouble() / 1000.0
        } catch (e: Exception) {
            Log.e(MainActivity.TAG, "Error getting audio duration", e)
            null
        }
    }

    suspend fun toggleRecording(shouldLoop: Boolean) {
        val currentTime = System.currentTimeMillis()
        if (isRecordingActionInProgress || (currentTime - lastRecordingActionTime < DEBOUNCE_TIMEOUT)) {
            Log.i(TAG, "Recording action in progress or debounce period active. Ignoring request.")
            return
        }

        try {
            isRecordingActionInProgress = true
            lastRecordingActionTime = currentTime

            if (isRecording.value) {
                val pcmBytes = audioRecorder?.getPcmData()
                Log.i(TAG, "Collected ${pcmBytes?.size ?: 0} bytes")
                if (pcmBytes != null) {
                    val seconds = pcmBytes.size / (AudioRecorder.SAMPLE_RATE * 2)
                    Log.i(TAG, "That is ~$seconds seconds of audio")
                }
                stopRecording()
                if (!shouldLoop && pcmBytes != null) {
                    appContext?.let { ctx ->
                        viewModelScope.launch {
                            startTranscription(ctx, pcmBytes)
                        }
                    }
                }
            } else {
                _errorMessage.value = null

                if (_modelState.value != ModelState.LOADED) {
                    _errorMessage.value = "Please wait for model to load completely before recording"
                    return
                }

                startRecording(shouldLoop)
            }
        } finally {
            isRecordingActionInProgress = false
        }
    }

    private fun startRecording(shouldLoop: Boolean) {
        if (_modelState.value != ModelState.LOADED) {
            _errorMessage.value = "Model is not fully loaded yet. Please wait."
            return
        }

        resetState()
        _isRecording.value = true
        _isInitializing.value = true

        viewModelScope.launch {
            val context = appContext

            val tempFile = File(
                context.cacheDir,
                "AUDIO_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.wav",
            )

            audioRecorder = AudioRecorder(context)
            if (!audioRecorder!!.startRecording(tempFile)) {
                Log.e(TAG, "Failed to start AudioRecorder")
                _isRecording.value = false
                _isInitializing.value = false
                _errorMessage.value = "Failed to start audio recording"
                audioRecorder = null
                return@launch
            }

            energyCollectJob?.cancel()
            energyCollectJob = launch {
                audioRecorder!!.bufferEnergy.collect { list ->
                    _bufferEnergy.clear()
                    _bufferEnergy.addAll(list)
                    _bufferSeconds.value = list.size * 0.1

                    if (_isInitializing.value && list.isNotEmpty()) {
                        _isInitializing.value = false
                    }
                }
            }
        }
    }

    private fun stopRecording() {
        if (!isRecording.value) return
        try {
            audioRecorder?.stopRecording()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recorder", e)
        }
        energyCollectJob?.cancel()
        audioRecorder = null
        _isRecording.value = false
    }

    private fun getRealPathFromUri(context: Context, uri: Uri?): String? {
        var filePath: String? = null

        if (uri == null) return null

        try {
            if (DocumentsContract.isDocumentUri(context, uri)) {
                val documentId = DocumentsContract.getDocumentId(uri)
                val split = documentId.split(":".toRegex()).dropLastWhile { it.isEmpty() }
                    .toTypedArray()

                if (split.isEmpty()) {
                    Log.e(TAG, "Invalid document ID format")
                    return null
                }

                val type = split[0]

                if ("primary" == type) {
                    val primaryPath = uri.path
                    filePath = primaryPath?.replace("/document/primary:", "/sdcard/")
                    return filePath
                } else if ("raw" == type) {
                    filePath = split[1]
                    Log.d(TAG, "Raw path: $filePath")
                    return filePath
                } else if ("audio" == type) {
                    val contentUri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI

                    val selection = "_id=?"
                    val selectionArgs = arrayOf(split[1])
                    try {
                        val cursor = context.contentResolver.query(
                            contentUri,
                            null,
                            selection,
                            selectionArgs,
                            null,
                        )

                        cursor?.use {
                            val columnIndex = it.getColumnIndex(MediaStore.Audio.Media.DATA)
                            if (columnIndex != -1 && it.moveToFirst()) {
                                filePath = it.getString(columnIndex)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error querying content resolver", e)
                    }
                }
            } else {
                val scheme = uri.scheme
                if (ContentResolver.SCHEME_CONTENT.equals(scheme, ignoreCase = true)) {
                    filePath = getDataColumn(context, uri, null, null)
                    val projection = arrayOf(MediaStore.MediaColumns.DATA)

                    try {
                        val cursor = context.contentResolver.query(uri, projection, null, null, null)
                        cursor?.use {
                            if (it.moveToFirst()) {
                                val columnIndex = it.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA)
                                filePath = it.getString(columnIndex)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error querying content resolver for path", e)
                    }

                    if (filePath == null) {
                        try {
                            val fileName = getFileNameFromUri(context, uri)
                            val extension = getFileExtensionFromUri(context, uri) ?: ".tmp"
                            val tempFile = File.createTempFile(
                                "audio_${System.currentTimeMillis()}",
                                extension,
                                context.cacheDir,
                            )

                            context.contentResolver.openInputStream(uri)?.use { input ->
                                FileOutputStream(tempFile).use { output ->
                                    input.copyTo(output)
                                }
                            }

                            filePath = tempFile.absolutePath
                            Log.d(TAG, "Created temp file: $filePath")
                        } catch (e: Exception) {
                            Log.e(TAG, "Error creating temp file from URI", e)
                        }
                    }
                } else if (ContentResolver.SCHEME_FILE.equals(scheme, ignoreCase = true)) {
                    filePath = uri.path
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting real path from URI", e)
        }

        return filePath
    }

    private fun getFileNameFromUri(context: Context, uri: Uri): String? {
        var fileName: String? = null
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val displayNameIndex = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (displayNameIndex != -1) {
                    fileName = it.getString(displayNameIndex)
                }
            }
        }
        return fileName
    }

    private fun getDataColumn(context: Context, uri: Uri, selection: String?, selectionArgs: Array<String>?): String? {
        val column = MediaStore.MediaColumns.DATA
        val projection = arrayOf(column)

        try {
            context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val columnIndex = cursor.getColumnIndexOrThrow(column)
                    return cursor.getString(columnIndex)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in getDataColumn", e)
        }

        return null
    }

    private fun getFileExtensionFromUri(context: Context, uri: Uri): String? {
        val fileName = getFileNameFromUri(context, uri) ?: return null
        val lastDotIndex = fileName.lastIndexOf(".")
        return if (lastDotIndex > 0) {
            fileName.substring(lastDotIndex)
        } else {
            null
        }
    }

    fun setTask(task: String) {
        _selectedTask.value = task
    }

    fun setLanguage(language: String) {
        _selectedLanguage.value = language
    }

    fun setEncoderComputeUnits(units: ComputeUnits) {
        _encoderComputeUnits.value = units
        if (_modelState.value == ModelState.LOADED) {
            _modelState.value = ModelState.LOADING
            _encoderState.value = ModelState.UNLOADED
            loadModel(_selectedModel.value)
        }
    }

    fun setDecoderComputeUnits(units: ComputeUnits) {
        _decoderComputeUnits.value = units
        if (_modelState.value == ModelState.LOADED) {
            _modelState.value = ModelState.LOADING
            _decoderState.value = ModelState.UNLOADED
            loadModel(_selectedModel.value)
        }
    }

    fun setSegmenterComputeUnits(units: ComputeUnits) {
        _segmenterComputeUnits.value = units
    }

    fun setEmbedderComputeUnits(units: ComputeUnits) {
        _embedderComputeUnits.value = units
    }

    private fun updateRealtimeMetrics(elapsedSeconds: Double) {
        if (latestTime > 0) {
            val processedAudioDuration = latestTime.toDouble()
            if (processedAudioDuration > 0) {
                _audioSampleDuration.value = processedAudioDuration
                _totalProcessTime.value = elapsedSeconds
                _effectiveRealTimeFactor.value = elapsedSeconds / processedAudioDuration
                _effectiveSpeedFactor.value = processedAudioDuration / elapsedSeconds
            }
        }
    }

    inner class DecoderCallbacks : AudioDecoderCallbacks {
        override fun onAudioFormat(freq: Int, ch: Int, dur: Long) {
            Log.i(MainActivity.TAG, "sample rate: $freq, channels: $ch, duration: $dur")
            samplerate = freq
            channels = ch
            duration = dur
        }

        override fun onOutputBuffer(pcmbuffer: ByteArray, timestamp: Long) {
            totalPCMdata += pcmbuffer.size
            if (timestamp > 0) {
                latestTime = (timestamp / 1000000).toInt()

                val elapsedSeconds = (System.currentTimeMillis() - startTime).toDouble() / 1000.0
                if (elapsedSeconds > 0) {
                    updateRealtimeMetrics(elapsedSeconds)
                }
            }

            if (!audioDecoder.isEndOfStream() && whisperKit.transcribe(pcmbuffer) < 30) {
                return
            }
        }

        override fun onDecodeClose() {
            try {
                Log.i(TAG, "Decoder closed, attempting to deinitialize WhisperKit")
                if (_isTranscribing.value) {
                    _isTranscribing.value = false
                    _isInitializing.value = false
                }

                try {
                    if (::whisperKit.isInitialized) {
                        whisperKit.deinitialize()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Error in deinitializing whisperkit: ${e.message}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in onDecodeClose", e)
            } finally {
                Log.i(TAG, "total data: $totalPCMdata, duration: $latestTime secs")
            }
        }

        override fun onEndOfStream() {
            // Not implemented
        }
    }

    fun stopTranscription() {
        viewModelScope.launch {
            Log.i(TAG, "Stopping transcription")

            withContext(Dispatchers.Main) {
                _isTranscribing.value = false
                _isInitializing.value = false
            }

            val elapsedSeconds = (System.currentTimeMillis() - startTime).toDouble() / 1000.0
            updateRealtimeMetrics(elapsedSeconds)

            try {
                if (::audioDecoder.isInitialized) {
                    audioDecoder.decodeStopRelease()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping audio decoder", e)
            }
            Log.i(TAG, "Transcription stopped")
        }
    }
}
