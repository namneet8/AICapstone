package com.example.hearmate.core.service

import android.Manifest
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.annotation.RequiresPermission
import androidx.core.app.NotificationCompat
import androidx.core.content.edit
import com.example.hearmate.MainActivity
import com.example.hearmate.R
import com.example.hearmate.core.audio.AudioRecorderManager
import com.example.hearmate.core.interfaces.SoundClassifier
import com.example.hearmate.core.interfaces.SoundDetectionResult
import com.example.hearmate.data.repository.SoundEventRepository
import com.example.hearmate.presentation.ui.activities.EmergencyAlertActivity
import dagger.hilt.android.AndroidEntryPoint
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import javax.inject.Inject
import kotlin.math.sqrt

/**
 * Foreground Service for continuous emergency sound detection.
 *
 * Architecture Overview:
 * ----------------------
 * This service runs in the background (even when app is closed) to:
 * 1. Record audio continuously via AudioRecorderManager
 * 2. Classify audio chunks using TFLite model (SoundClassifier)
 * 3. Trigger emergency alerts when emergency sounds are detected
 * 4. Manage pause/resume state with persistence
 *
 * State Persistence:
 * - Listening enabled state and pause state are saved to SharedPreferences
 * - If app is killed while paused, service restores pause timer on restart
 * - If pause expires while app is closed, listening auto-resumes
 *
 * Thread Safety:
 * - TFLite interpreter is NOT thread-safe
 * - All classifier calls are serialized using a Mutex
 */
@RequiresApi(Build.VERSION_CODES.O)
@AndroidEntryPoint
class ListeningService : Service() {

    companion object {
        private const val TAG = "ListeningService"

        // Notification IDs
        private const val SERVICE_NOTIFICATION_ID = 1
        private const val EMERGENCY_NOTIFICATION_ID = 999

        // Notification Channel IDs
        private const val SERVICE_CHANNEL_ID = "listening_channel"
        private const val EMERGENCY_CHANNEL_ID = "emergency_channel"

        // SharedPreferences for service state persistence
        private const val SERVICE_PREFS_NAME = "hearmate_service_prefs"
        private const val KEY_IS_LISTENING_ENABLED = "is_listening_enabled"
        private const val KEY_IS_PAUSED = "is_paused"
        private const val KEY_PAUSE_END_TIME = "pause_end_time"

        // SharedPreferences for alert settings (shared with ViewModel)
        private const val ALERT_PREFS_NAME = "hearmate_vibration_prefs"
    }

    // ========================================
    // Injected Dependencies
    // ========================================

    @Inject lateinit var audioRecorderManager: AudioRecorderManager
    @Inject lateinit var soundClassifier: SoundClassifier
    @Inject lateinit var repository: SoundEventRepository

    // ========================================
    // System Services & Preferences
    // ========================================

    private lateinit var vibrator: Vibrator
    private lateinit var servicePrefs: SharedPreferences
    private lateinit var alertPrefs: SharedPreferences
    private lateinit var notificationManager: NotificationManager

    // ========================================
    // Coroutines & Threading
    // ========================================

    private val mainHandler = Handler(Looper.getMainLooper())
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Mutex ensures thread-safe access to TFLite interpreter
    // TFLite crashes (SIGBUS/Scudo) if accessed from multiple threads simultaneously
    private val classifierMutex = Mutex()

    // Job for pause countdown timer
    private var pauseTimerJob: Job? = null

    // ========================================
    // Exposed State Flows (observed by ServiceManager)
    // ========================================

    /** True when audio is actively being recorded and classified */
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    /** True when user has enabled listening (toggle ON), may still be paused */
    private val _isListeningEnabled = MutableStateFlow(false)
    val isListeningEnabled: StateFlow<Boolean> = _isListeningEnabled.asStateFlow()

    /** True when monitoring is temporarily paused */
    private val _isPaused = MutableStateFlow(false)
    val isPaused: StateFlow<Boolean> = _isPaused.asStateFlow()

    /** Seconds remaining until pause expires and listening auto-resumes */
    private val _pauseTimeRemaining = MutableStateFlow(0L)
    val pauseTimeRemaining: StateFlow<Long> = _pauseTimeRemaining.asStateFlow()

    // ========================================
    // Service Lifecycle
    // ========================================

    override fun onCreate() {
        super.onCreate()

        // Initialize vibrator (handles API level differences)
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            getSystemService(Vibrator::class.java)
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // Initialize SharedPreferences
        servicePrefs = getSharedPreferences(SERVICE_PREFS_NAME, Context.MODE_PRIVATE)
        alertPrefs = getSharedPreferences(ALERT_PREFS_NAME, Context.MODE_PRIVATE)
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // Restore state from previous session (handles app restart)
        restorePersistedState()

        Log.d(TAG, "Service created - restored: enabled=${_isListeningEnabled.value}, paused=${_isPaused.value}")
    }

    override fun onBind(intent: Intent): IBinder = LocalBinder()

    @SuppressLint("MissingPermission")
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Create notification channels (required for Android 8+)
        createNotificationChannels()

        // Start as foreground service with persistent notification
        startForeground(SERVICE_NOTIFICATION_ID, createServiceNotification())

        // Auto-resume monitoring if it was enabled and not paused
        // Permission was already granted when user first enabled listening
        if (_isListeningEnabled.value && !_isPaused.value) {
            startAudioMonitoring()
        }

        Log.d(TAG, "Service started in foreground")

        // START_STICKY: System will restart service if killed
        return START_STICKY
    }

    override fun onDestroy() {
        stopAudioMonitoring()
        pauseTimerJob?.cancel()
        serviceScope.cancel()
        // Note: Don't call audioRecorderManager.onDestroy() here!
        // AudioRecorderManager is a @Singleton that outlives this service.
        super.onDestroy()
    }

    // ========================================
    // State Persistence
    // ========================================

    /**
     * Restores service state from SharedPreferences.
     * Called on service creation to recover from app restart.
     */
    private fun restorePersistedState() {
        _isListeningEnabled.value = servicePrefs.getBoolean(KEY_IS_LISTENING_ENABLED, false)

        val wasPaused = servicePrefs.getBoolean(KEY_IS_PAUSED, false)
        val pauseEndTime = servicePrefs.getLong(KEY_PAUSE_END_TIME, 0L)

        if (wasPaused && pauseEndTime > 0) {
            val remainingMs = pauseEndTime - System.currentTimeMillis()

            if (remainingMs > 0) {
                // Pause is still active - restore the countdown timer
                _isPaused.value = true
                _pauseTimeRemaining.value = remainingMs / 1000
                startPauseCountdown(remainingMs)
                Log.d(TAG, "Restored pause: ${remainingMs / 1000}s remaining")
            } else {
                // Pause expired while app was closed - clear pause state
                clearPauseState()
                Log.d(TAG, "Pause expired while closed - will resume listening")
            }
        }
    }

    /**
     * Saves current listening enabled state to SharedPreferences.
     */
    private fun persistListeningState() {
        servicePrefs.edit {
            putBoolean(KEY_IS_LISTENING_ENABLED, _isListeningEnabled.value)
        }
    }

    /**
     * Saves pause end timestamp for recovery after restart.
     */
    private fun persistPauseState(pauseEndTimeMs: Long) {
        servicePrefs.edit {
            putBoolean(KEY_IS_PAUSED, true)
            putLong(KEY_PAUSE_END_TIME, pauseEndTimeMs)
        }
    }

    /**
     * Clears pause state from memory and SharedPreferences.
     */
    private fun clearPauseState() {
        _isPaused.value = false
        _pauseTimeRemaining.value = 0L
        pauseTimerJob?.cancel()
        pauseTimerJob = null

        servicePrefs.edit {
            putBoolean(KEY_IS_PAUSED, false)
            putLong(KEY_PAUSE_END_TIME, 0L)
        }
    }

    // ========================================
    // Public API (called by ServiceManager)
    // ========================================

    /**
     * Enables listening and starts audio monitoring.
     * Called when user turns the toggle ON.
     *
     * @return true if monitoring started successfully
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun enableListening(): Boolean {
        if (_isListeningEnabled.value && _isListening.value) {
            Log.d(TAG, "Already enabled and listening")
            return true
        }

        _isListeningEnabled.value = true
        persistListeningState()

        // Only start monitoring if not currently paused
        if (!_isPaused.value) {
            return startAudioMonitoring()
        }

        Log.d(TAG, "Listening enabled but currently paused")
        return true
    }

    /**
     * Disables listening completely.
     * Called when user turns the toggle OFF.
     */
    fun disableListening() {
        stopAudioMonitoring()
        clearPauseState()

        _isListeningEnabled.value = false
        persistListeningState()

        updateServiceNotification("Service active (monitoring OFF)")
        Log.d(TAG, "Listening disabled")
    }

    /**
     * Pauses monitoring for specified duration.
     * Timer runs in background and auto-resumes when expired.
     *
     * @param durationMinutes Pause duration in minutes
     */
    fun pauseListening(durationMinutes: Int) {
        if (!_isListeningEnabled.value) {
            Log.w(TAG, "Cannot pause - listening not enabled")
            return
        }

        // Stop audio recording
        stopAudioMonitoring()

        // Calculate pause end timestamp
        val durationMs = durationMinutes * 60 * 1000L
        val pauseEndTime = System.currentTimeMillis() + durationMs

        // Update state
        _isPaused.value = true
        _pauseTimeRemaining.value = durationMinutes * 60L

        // Persist for recovery after restart
        persistPauseState(pauseEndTime)

        // Start countdown timer
        startPauseCountdown(durationMs)

        updateServiceNotification("Paused for $durationMinutes minutes")
        Log.d(TAG, "Paused for $durationMinutes minutes (until $pauseEndTime)")
    }

    /**
     * Resumes monitoring immediately.
     * Called when user clicks "Resume Now" or when pause timer expires.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun resumeListening() {
        clearPauseState()

        if (_isListeningEnabled.value) {
            startAudioMonitoring()
            Log.d(TAG, "Resumed listening")
        } else {
            Log.d(TAG, "Resume called but listening not enabled")
        }
    }

    // ========================================
    // Pause Timer
    // ========================================

    /**
     * Starts countdown timer for pause duration.
     * Updates pauseTimeRemaining every second.
     * Auto-calls resumeListening() when timer expires.
     */
    @SuppressLint("MissingPermission")
    private fun startPauseCountdown(durationMs: Long) {
        pauseTimerJob?.cancel()

        pauseTimerJob = serviceScope.launch {
            var remainingMs = durationMs

            while (remainingMs > 0 && isActive) {
                _pauseTimeRemaining.value = remainingMs / 1000
                delay(1000L)
                remainingMs -= 1000L
            }

            if (isActive) {
                // Timer expired - resume on main thread
                withContext(Dispatchers.Main) {
                    resumeListening()
                }
            }
        }
    }

    // ========================================
    // Audio Monitoring
    // ========================================

    /**
     * Starts audio recording and classification pipeline.
     *
     * Flow:
     * 1. AudioRecorderManager starts recording from microphone
     * 2. When audio chunk is ready (2 seconds), it's emitted via audioData flow
     * 3. We collect chunks and pass them to the classifier
     * 4. If emergency sound detected, trigger alert
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startAudioMonitoring(): Boolean {
        if (_isListening.value) return true

        return try {
            val started = audioRecorderManager.startRecording()

            if (started) {
                _isListening.value = true

                // Collect audio chunks and classify them
                serviceScope.launch {
                    audioRecorderManager.audioData.collect { chunk ->
                        chunk?.let { processAudioChunk(it) }
                    }
                }

                updateServiceNotification("Monitoring for emergency sounds...")
                Log.d(TAG, "Audio monitoring started")
                true
            } else {
                Log.e(TAG, "Failed to start audio recording")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error starting audio monitoring: ${e.message}", e)
            false
        }
    }

    /**
     * Stops audio recording.
     */
    private fun stopAudioMonitoring() {
        audioRecorderManager.stopRecording()
        _isListening.value = false
        Log.d(TAG, "Audio monitoring stopped")
    }

    /**
     * Processes a 2-second audio chunk through the classifier.
     *
     * @param audioChunk Raw audio samples (FloatArray)
     */
    private suspend fun processAudioChunk(audioChunk: FloatArray) {
        try {
            // Use mutex to ensure thread-safe TFLite access
            val result = classifierMutex.withLock {
                soundClassifier.classifySound(audioChunk)
            }

            val isEmergency = soundClassifier.isEmergencySound(result.label)
            val rmsLevel = calculateRMS(audioChunk)

            // Save event to local database (for history)
            withContext(Dispatchers.IO) {
                try {
                    repository.saveEvent(
                        result = result,
                        rmsLevel = rmsLevel,
                        isEmergency = isEmergency
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to save event: ${e.message}")
                }
            }

            // Trigger alert for emergency sounds
            if (isEmergency) {
                mainHandler.post { handleEmergencySound(result) }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing audio: ${e.message}", e)
        }
    }

    /**
     * Calculates Root Mean Square (RMS) of audio samples.
     * RMS indicates the overall loudness/energy of the audio.
     */
    private fun calculateRMS(buffer: FloatArray): Double {
        if (buffer.isEmpty()) return 0.0
        val sumSquares = buffer.sumOf { (it * it).toDouble() }
        return sqrt(sumSquares / buffer.size)
    }

    // ========================================
    // Emergency Alert Handling
    // ========================================

    /**
     * Handles detected emergency sound.
     * Checks if alerts are enabled for this sound type.
     * If enabled, triggers vibration and shows alert UI.
     */
    private fun handleEmergencySound(result: SoundDetectionResult) {
        // Check if alerts are enabled for this sound type
        val alertsEnabled = alertPrefs.getBoolean(result.label, true)
        if (!alertsEnabled) {
            Log.d(TAG, "Alerts disabled for ${result.label}, skipping")
            return
        }

        Log.w(TAG, "EMERGENCY DETECTED: ${result.label}")

        // Trigger vibration
        triggerVibration()

        // Show emergency alert (Activity or notification depending on app state)
        launchEmergencyAlert(result)

        // Update persistent notification
        updateServiceNotification("Emergency detected: ${result.label}")
    }

    /**
     * Triggers emergency vibration pattern.
     * Pattern: vibrate 300ms, pause 100ms, repeat 3 times
     */
    private fun triggerVibration() {
        try {
            if (!vibrator.hasVibrator()) return

            val pattern = longArrayOf(0, 300, 100, 300, 100, 300)
            val effect = VibrationEffect.createWaveform(pattern, -1)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                vibrator.vibrate(
                    effect,
                    android.os.VibrationAttributes.Builder()
                        .setUsage(android.os.VibrationAttributes.USAGE_ALARM)
                        .build()
                )
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(effect)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Vibration failed: ${e.message}")
        }
    }

    /**
     * Launches emergency alert UI.
     *
     * Strategy:
     * - If app is in foreground: Launch EmergencyAlertActivity directly
     * - If app is in background: Show high-priority notification with fullScreenIntent
     *   (Android 10+ blocks startActivity from background, but fullScreenIntent works)
     */
    @SuppressLint("FullScreenIntentPolicy")
    private fun launchEmergencyAlert(result: SoundDetectionResult) {
        try {
            val intent = Intent(this, EmergencyAlertActivity::class.java).apply {
                putExtra(EmergencyAlertActivity.EXTRA_SOUND_LABEL, result.label)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_CLEAR_TOP or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP
            }

            if (isAppInForeground()) {
                // App visible - launch Activity directly
                startActivity(intent)
                Log.d(TAG, "App in foreground - launched Activity directly")
            } else {
                // App in background - use notification with fullScreenIntent
                showEmergencyNotification(result, intent)
                Log.d(TAG, "App in background - notification with fullScreenIntent sent")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch alert: ${e.message}")
        }
    }

    /**
     * Shows high-priority notification for emergency when app is in background.
     * Uses fullScreenIntent to wake screen and show alert.
     */
    private fun showEmergencyNotification(result: SoundDetectionResult, alertIntent: Intent) {
        val fullScreenIntent = PendingIntent.getActivity(
            this,
            System.currentTimeMillis().toInt(),
            alertIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val canUseFullScreen = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            notificationManager.canUseFullScreenIntent()
        } else {
            true
        }

        val notification = NotificationCompat.Builder(this, EMERGENCY_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("HearMate detected \"${result.label}\" sound")
            .setContentText("Emergency sound detected. Tap to view details.")
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setAutoCancel(true)
            .setContentIntent(fullScreenIntent)
            .setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM))
            .setDefaults(NotificationCompat.DEFAULT_VIBRATE or NotificationCompat.DEFAULT_LIGHTS)
            .apply {
                if (canUseFullScreen) {
                    setFullScreenIntent(fullScreenIntent, true)
                }
            }
            .build()

        notificationManager.notify(EMERGENCY_NOTIFICATION_ID, notification)
    }

    /**
     * Checks if the app is currently visible to the user.
     */
    private fun isAppInForeground(): Boolean {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val appProcesses = activityManager.runningAppProcesses ?: return false

        return appProcesses.any { process ->
            process.importance == ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND &&
                    process.processName == packageName
        }
    }

    // ========================================
    // Notifications
    // ========================================

    /**
     * Creates notification channels (required for Android 8+).
     */
    private fun createNotificationChannels() {
        // Low-priority channel for persistent service notification
        val serviceChannel = NotificationChannel(
            SERVICE_CHANNEL_ID,
            "HearMate Monitoring",
            NotificationManager.IMPORTANCE_LOW
        )

        // High-priority channel for emergency alerts
        val emergencyChannel = NotificationChannel(
            EMERGENCY_CHANNEL_ID,
            "Emergency Alerts",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Critical alerts for emergency sounds"
            enableVibration(true)
            lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            setSound(
                RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM),
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ALARM)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
        }

        notificationManager.createNotificationChannel(serviceChannel)
        notificationManager.createNotificationChannel(emergencyChannel)
    }

    /**
     * Creates the persistent foreground service notification.
     */
    private fun createServiceNotification(text: String = "Service active"): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, SERVICE_CHANNEL_ID)
            .setContentTitle("HearMate")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    /**
     * Updates the persistent notification text.
     */
    private fun updateServiceNotification(text: String) {
        val notification = createServiceNotification(text)
        notificationManager.notify(SERVICE_NOTIFICATION_ID, notification)
    }

    // ========================================
    // Binder for Service Binding
    // ========================================

    inner class LocalBinder : Binder() {
        fun getService(): ListeningService = this@ListeningService
    }
}