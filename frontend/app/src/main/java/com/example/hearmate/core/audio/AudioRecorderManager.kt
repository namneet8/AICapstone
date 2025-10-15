package com.example.hearmate.core.audio

import android.Manifest
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

@Singleton
class AudioRecorderManager @Inject constructor(
    @ApplicationContext private val context: Context
) {

    companion object {
        private const val TAG = "AudioRecorder"

        // Audio settings
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // Threshold to trigger recording
        private const val RMS_THRESHOLD = 0.01

        // Recording duration when triggered
        private const val RECORDING_DURATION_MS = 2000L // 2 seconds
        private const val SAMPLES_TO_RECORD = (SAMPLE_RATE * RECORDING_DURATION_MS / 1000).toInt()
    }

    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null

    private val bufferSize = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    ) * 4

    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    // True only when actively capturing 2 seconds after threshold triggered
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    // Emits 2-second chunks only when sound detected
    private val _audioData = MutableStateFlow<FloatArray?>(null)
    val audioData: StateFlow<FloatArray?> = _audioData.asStateFlow()

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording(): Boolean {
        if (_isListening.value) return true

        return try {
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
                    if (state != AudioRecord.STATE_INITIALIZED) {
                        Log.e(TAG, "AudioRecord initialization failed")
                        release()
                        return false
                    }
                    startRecording()
                }

            _isListening.value = true
            startRecordingCoroutine()
            Log.d(TAG, "Started - Threshold: $RMS_THRESHOLD")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            false
        }
    }

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
        _isRecording.value = false
    }

    private fun startRecordingCoroutine() {
        recordingJob = scope.launch(Dispatchers.Default) {
            val monitorBuffer = ShortArray(bufferSize / 4) // Small buffer for monitoring

            while (isActive && _isListening.value) {
                // PHASE 1: MONITORING - wait for sound above threshold
                var soundDetected = false

                while (isActive && !soundDetected) {
                    val readBytes = audioRecord?.read(monitorBuffer, 0, monitorBuffer.size) ?: 0

                    if (readBytes > 0) {
                        val floatBuffer = FloatArray(readBytes) { i -> monitorBuffer[i] / 32768.0f }
                        val rms = calculateRMS(floatBuffer)

                        if (rms >= RMS_THRESHOLD) {
                            Log.d(TAG, "Threshold triggered! RMS: %.4f - Starting 2s recording".format(rms))
                            soundDetected = true
                        }
                    }
                }

                if (!isActive) break

                // PHASE 2: RECORDING - capture exactly 2 seconds
                _isRecording.value = true //set true for recording
                Log.d(TAG, "Recording 2 seconds...")
                val recordedChunk = mutableListOf<Float>()

                while (isActive && recordedChunk.size < SAMPLES_TO_RECORD) {
                    val readBytes = audioRecord?.read(monitorBuffer, 0, monitorBuffer.size) ?: 0

                    if (readBytes > 0) {
                        for (i in 0 until readBytes) {
                            recordedChunk.add(monitorBuffer[i] / 32768.0f)
                        }
                    }
                }

                if (!isActive) break

                // PHASE 3: EMIT - send to classifier
                val finalChunk = recordedChunk.take(SAMPLES_TO_RECORD).toFloatArray()
                val finalRMS = calculateRMS(finalChunk)
                Log.d(TAG, "Emitting chunk - RMS: %.4f, Samples: %d".format(finalRMS, finalChunk.size))
                _audioData.emit(finalChunk)

                _isRecording.value = false // set to false after recording

                // Small pause before next monitoring cycle
                delay(500)
            }
        }
    }

    private fun calculateRMS(buffer: FloatArray): Double {
        if (buffer.isEmpty()) return 0.0
        val sumSquares = buffer.sumOf { (it * it).toDouble() }
        return sqrt(sumSquares / buffer.size)
    }

    fun onDestroy() {
        stopRecording()
        scope.cancel()
    }
}