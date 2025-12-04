package com.example.hearmate.core.audio

import android.Manifest
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

/**
 * Manages audio recording from the device microphone.
 *
 * Architecture Overview:
 * ----------------------
 * This class handles continuous audio monitoring with a threshold-based approach:
 *
 * 1. MONITORING PHASE: Continuously reads small audio buffers
 *    - Calculates RMS (volume level) of each buffer
 *    - If RMS exceeds threshold, triggers recording phase
 *
 * 2. RECORDING PHASE: Captures exactly 2 seconds of audio
 *    - Records 32,000 samples at 16kHz sample rate
 *    - Emits the audio chunk via audioData StateFlow
 *
 * 3. Returns to monitoring phase after a brief pause
 *
 * Why threshold-based?
 * - Saves battery by not continuously processing silence
 * - Model only needs to classify when there's actual sound
 * - 2-second chunks are optimal for the TFLite model
 *
 * Thread Safety:
 * - Uses coroutines for background processing
 * - StateFlows are thread-safe for observation
 *
 * Note: This is a @Singleton because it manages hardware resources
 * that should only be acquired once.
 */
@Singleton
class AudioRecorderManager @Inject constructor(
    @ApplicationContext private val context: Context
) {

    companion object {
        private const val TAG = "AudioRecorder"

        // Audio format settings (must match model requirements)
        private const val SAMPLE_RATE = 16000  // 16kHz sample rate
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // Threshold for triggering recording (RMS value)
        // Lower = more sensitive, Higher = less sensitive
        private const val RMS_THRESHOLD = 0.01

        // Recording duration when threshold is triggered
        private const val RECORDING_DURATION_MS = 2000L  // 2 seconds
        private const val SAMPLES_TO_RECORD = (SAMPLE_RATE * RECORDING_DURATION_MS / 1000).toInt()
        // At 16kHz, 2 seconds = 32,000 samples
    }

    // ========================================
    // State
    // ========================================

    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null

    // Buffer size for AudioRecord (minimum + extra for safety)
    private val bufferSize = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    ) * 4

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // ========================================
    // Exposed State Flows
    // ========================================

    /** Whether audio monitoring is currently active */
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    /**
     * Emits 2-second audio chunks when sound is detected.
     * Observers (ListeningService) collect this to pass to the classifier.
     *
     * Value is null when no new audio is available.
     */
    private val _audioData = MutableStateFlow<FloatArray?>(null)
    val audioData: StateFlow<FloatArray?> = _audioData.asStateFlow()

    // ========================================
    // Public API
    // ========================================

    /**
     * Starts audio monitoring.
     *
     * Flow:
     * 1. Creates and initializes AudioRecord
     * 2. Starts the monitoring coroutine
     * 3. Returns true if successful
     *
     * @return true if recording started successfully, false otherwise
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording(): Boolean {
        if (_isListening.value) return true  // Already running

        return try {
            // Create AudioRecord with specified format
            audioRecord = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AUDIO_FORMAT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(CHANNEL_CONFIG)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .build()
                .apply {
                    // Check if initialization succeeded
                    if (state != AudioRecord.STATE_INITIALIZED) {
                        Log.e(TAG, "AudioRecord initialization failed")
                        release()
                        return false
                    }
                    startRecording()
                }

            _isListening.value = true
            startMonitoringCoroutine()
            Log.d(TAG, "Started - Threshold: $RMS_THRESHOLD")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            false
        }
    }

    /**
     * Stops audio monitoring and releases resources.
     */
    fun stopRecording() {
        recordingJob?.cancel()
        recordingJob = null

        audioRecord?.let {
            if (it.state == AudioRecord.STATE_INITIALIZED) {
                it.stop()
                it.release()
            }
        }
        audioRecord = null
        _isListening.value = false
    }

    /**
     * Cleans up all resources. Call when app is being destroyed.
     */
    fun onDestroy() {
        stopRecording()
        scope.cancel()
    }

    // ========================================
    // Internal - Monitoring Loop
    // ========================================

    /**
     * Main audio monitoring coroutine.
     *
     * Continuously cycles between:
     * - Monitoring phase: Wait for sound above threshold
     * - Recording phase: Capture 2 seconds of audio
     * - Emit phase: Send audio chunk to observers
     */
    private fun startMonitoringCoroutine() {
        recordingJob = scope.launch(Dispatchers.Default) {
            // Small buffer for monitoring phase
            val monitorBuffer = ShortArray(bufferSize / 4)

            while (isActive && _isListening.value) {
                // ----------------------------------------
                // PHASE 1: MONITORING - Wait for sound
                // ----------------------------------------
                var soundDetected = false

                while (isActive && !soundDetected) {
                    val samplesRead = audioRecord?.read(monitorBuffer, 0, monitorBuffer.size) ?: 0

                    if (samplesRead > 0) {
                        // Convert to float and calculate RMS
                        val floatBuffer = FloatArray(samplesRead) { i ->
                            monitorBuffer[i] / 32768.0f  // Normalize to [-1, 1]
                        }
                        val rms = calculateRMS(floatBuffer)

                        if (rms >= RMS_THRESHOLD) {
                            Log.d(TAG, "Threshold triggered! RMS: %.4f - Starting 2s recording".format(rms))
                            soundDetected = true
                        }
                    }
                }

                if (!isActive) break

                // ----------------------------------------
                // PHASE 2: RECORDING - Capture 2 seconds
                // ----------------------------------------
                Log.d(TAG, "Recording 2 seconds...")
                val recordedChunk = mutableListOf<Float>()

                while (isActive && recordedChunk.size < SAMPLES_TO_RECORD) {
                    val samplesRead = audioRecord?.read(monitorBuffer, 0, monitorBuffer.size) ?: 0

                    if (samplesRead > 0) {
                        for (i in 0 until samplesRead) {
                            recordedChunk.add(monitorBuffer[i] / 32768.0f)
                        }
                    }
                }

                if (!isActive) break

                // ----------------------------------------
                // PHASE 3: EMIT - Send to classifier
                // ----------------------------------------
                val finalChunk = recordedChunk.take(SAMPLES_TO_RECORD).toFloatArray()
                val finalRMS = calculateRMS(finalChunk)
                Log.d(TAG, "Emitting chunk - RMS: %.4f, Samples: %d".format(finalRMS, finalChunk.size))

                _audioData.emit(finalChunk)

                // Brief pause before next monitoring cycle
                delay(500)
            }
        }
    }

    /**
     * Calculates Root Mean Square (RMS) of audio samples.
     * RMS represents the average energy/loudness of the audio.
     *
     * Formula: sqrt(sum(x^2) / n)
     *
     * @param buffer Audio samples (normalized to [-1, 1])
     * @return RMS value (0.0 to 1.0, typically much lower)
     */
    private fun calculateRMS(buffer: FloatArray): Double {
        if (buffer.isEmpty()) return 0.0
        val sumSquares = buffer.sumOf { (it * it).toDouble() }
        return sqrt(sumSquares / buffer.size)
    }
}