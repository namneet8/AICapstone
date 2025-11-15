package com.example.hearmate.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.hearmate.core.audio.AudioRecorderManager
import com.example.hearmate.core.audio.SoundClassifier
import com.example.hearmate.core.audio.SoundDetectionResult
import com.example.hearmate.data.repository.SoundEventRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import kotlin.math.sqrt

@HiltViewModel
class ListeningViewModel @Inject constructor(
    private val audioRecorderManager: AudioRecorderManager,
    private val soundClassifier: SoundClassifier,
    val repository: SoundEventRepository  // Exposed for SettingsScreen
) : ViewModel() {

    // Listening state (true when app is listening)
    val isListening: StateFlow<Boolean> = audioRecorderManager.isListening

    // Recording state (true only when recording 2 seconds after threshold)
    val isThresholdTriggered: StateFlow<Boolean> = audioRecorderManager.isRecording

    // Last detected sound
    private val _lastDetectedSound = MutableStateFlow<SoundDetectionResult?>(null)
    val lastDetectedSound: StateFlow<SoundDetectionResult?> = _lastDetectedSound.asStateFlow()

    // Current RMS level for UI
    private val _currentRMSLevel = MutableStateFlow(0.0)
    val currentRMSLevel: StateFlow<Double> = _currentRMSLevel.asStateFlow()

    // Emergency alert (triggers red flashing screen)
    private val _isEmergencyAlert = MutableStateFlow(false)
    val isEmergencyAlert: StateFlow<Boolean> = _isEmergencyAlert.asStateFlow()

    private val _emergencySound = MutableStateFlow<SoundDetectionResult?>(null)
    val emergencySound: StateFlow<SoundDetectionResult?> = _emergencySound.asStateFlow()

    // Sync status
    private val _syncStatus = MutableStateFlow<String?>(null)
    val syncStatus: StateFlow<String?> = _syncStatus.asStateFlow()

    init {
        // Listen to audio data from recorder
        viewModelScope.launch {
            audioRecorderManager.audioData.collect { audioData ->
                audioData?.let {
                    // Calculate RMS for display
                    val rms = calculateRMS(it)
                    _currentRMSLevel.value = rms

                    // Classify the sound
                    android.util.Log.d("ListeningViewModel", "Classifying sound - RMS: %.4f".format(rms))
                    processAudioChunk(it, rms)
                }
            }
        }
    }

    // Start audio recording
    fun startListening(): Boolean {
        return try {
            audioRecorderManager.startRecording()
        } catch (e: SecurityException) {
            false
        }
    }

    // Stop audio recording
    fun stopListening() {
        audioRecorderManager.stopRecording()
    }

    // Process audio chunk with sound classifier
    private suspend fun processAudioChunk(audioChunk: FloatArray, rmsLevel: Double) {
        try {
            val result = soundClassifier.classifySound(audioChunk)
            _lastDetectedSound.value = result

            // Check if emergency sound detected
            val isEmergency = soundClassifier.isEmergencySound(result.label)

            // Save to database automatically
            viewModelScope.launch {
                try {
                    val eventId = repository.saveEvent(
                        result = result,
                        rmsLevel = rmsLevel,
                        isEmergency = isEmergency
                    )
                    android.util.Log.d("ListeningViewModel", "Saved event #$eventId to database")
                } catch (e: Exception) {
                    android.util.Log.e("ListeningViewModel", "Failed to save event: ${e.message}")
                }
            }

            if (isEmergency) {
                handleEmergencySound(result)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Handle emergency sound detection
    private fun handleEmergencySound(result: SoundDetectionResult) {
        android.util.Log.e("ListeningViewModel", "DETECTED EMERGENCY: ${result.label} (${result.confidence * 100}%)")
        _emergencySound.value = result

        // Trigger red flashing alert
        _isEmergencyAlert.value = true
    }

    // Dismiss emergency alert manually
    fun dismissEmergencyAlert() {
        _isEmergencyAlert.value = false
    }

    // Sync events to MongoDB
    fun syncToMongoDB() {
        viewModelScope.launch {
            _syncStatus.value = "Syncing..."

            val (successCount, failureCount) = repository.syncToMongoDB()

            when {
                failureCount == -1 -> {
                    _syncStatus.value = "Network error - check connection"
                }
                failureCount > 0 -> {
                    _syncStatus.value = "Synced $successCount, failed $failureCount"
                }
                successCount > 0 -> {
                    _syncStatus.value = "Successfully synced $successCount events!"
                }
                else -> {
                    _syncStatus.value = "No events to sync"
                }
            }

            // Clear status after 3 seconds
            kotlinx.coroutines.delay(3000)
            _syncStatus.value = null
        }
    }

    // Get unsynced events count
    suspend fun getUnsyncedCount(): Int {
        return repository.getUnsyncedCount()
    }

    // Calculate RMS (Root Mean Square) for audio level
    private fun calculateRMS(buffer: FloatArray): Double {
        if (buffer.isEmpty()) return 0.0
        val sumSquares = buffer.sumOf { (it * it).toDouble() }
        return sqrt(sumSquares / buffer.size)
    }

    override fun onCleared() {
        super.onCleared()
        audioRecorderManager.onDestroy()
    }
}