package com.example.hearmate.presentation.viewmodel

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.content.edit
import androidx.lifecycle.ViewModel
import com.example.hearmate.core.service.ServiceManager
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

/**
 * ViewModel for the ListeningScreen UI.
 *
 * Architecture Overview:
 * ----------------------
 * This ViewModel acts as a thin layer between the UI (ListeningScreen) and the
 * ServiceManager. It proxies service state to the UI and manages user preferences.
 *
 * Responsibilities:
 * - Proxy service states (listening, paused, timer) to UI via StateFlows
 * - Manage vibration preferences for each emergency sound type
 * - Delegate start/stop/pause/resume actions to ServiceManager
 *
 * Note: All pause/timer logic is handled by ListeningService.
 * This ViewModel does NOT own any audio or detection logic.
 */
@RequiresApi(Build.VERSION_CODES.O)
@HiltViewModel
class ListeningViewModel @Inject constructor(
    private val serviceManager: ServiceManager,
    @ApplicationContext private val context: Context
) : ViewModel() {

    companion object {
        private const val TAG = "ListeningViewModel"
        private const val PREFS_NAME = "hearmate_vibration_prefs"

        /**
         * List of emergency sounds the app can detect.
         * These labels must match the TFLite model's output labels exactly.
         */
        private val EMERGENCY_SOUNDS = listOf(
            "Alarm_Clock",
            "Car_Horn",
            "Glass_Breaking",
            "Gunshot",
            "Siren"
        )
    }

    // ========================================
    // Vibration Preferences
    // ========================================

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // Map of sound label -> vibration enabled (true/false)
    private val _vibrationPreferences = MutableStateFlow<Map<String, Boolean>>(emptyMap())
    val vibrationPreferences: StateFlow<Map<String, Boolean>> = _vibrationPreferences.asStateFlow()

    // ========================================
    // Service State (Proxied from ServiceManager)
    // ========================================

    /** Whether audio is actively being recorded and classified */
    val isListening: StateFlow<Boolean> = serviceManager.isListening

    /** Whether user has turned the listening toggle ON (may still be paused) */
    val isListeningEnabled: StateFlow<Boolean> = serviceManager.isListeningEnabled

    /** Whether monitoring is temporarily paused */
    val isPaused: StateFlow<Boolean> = serviceManager.isPaused

    /** Seconds remaining until pause auto-expires */
    val pauseTimeRemaining: StateFlow<Long> = serviceManager.pauseTimeRemaining

    // ========================================
    // Initialization
    // ========================================

    init {
        loadVibrationPreferences()
        // Bind to service to sync current state (service may already be running)
        serviceManager.bind()
    }

    /**
     * Loads saved vibration preferences from SharedPreferences.
     * Defaults to enabled (true) for all sounds if not previously set.
     */
    private fun loadVibrationPreferences() {
        val prefs = EMERGENCY_SOUNDS.associateWith { sound ->
            sharedPreferences.getBoolean(sound, true) // Default: vibration ON
        }
        _vibrationPreferences.value = prefs
    }

    // ========================================
    // Public API - Settings
    // ========================================

    /**
     * Returns list of all detectable emergency sound labels.
     * Used by SettingsScreen to display vibration toggles.
     */
    fun getEmergencySounds(): List<String> = EMERGENCY_SOUNDS

    /**
     * Updates vibration preference for a specific sound type.
     *
     * @param soundLabel The sound label (e.g., "Siren", "Gunshot")
     * @param enabled Whether to vibrate when this sound is detected
     */
    fun setVibrationEnabled(soundLabel: String, enabled: Boolean) {
        // Persist to SharedPreferences
        sharedPreferences.edit { putBoolean(soundLabel, enabled) }

        // Update in-memory state
        _vibrationPreferences.value = _vibrationPreferences.value.toMutableMap().apply {
            this[soundLabel] = enabled
        }

        Log.d(TAG, "Vibration for $soundLabel set to $enabled")
    }

    // ========================================
    // Public API - Listening Control
    // ========================================

    /**
     * Starts audio monitoring (turns toggle ON).
     *
     * Flow:
     * 1. Starts the foreground service if not running
     * 2. Binds to service for state updates
     * 3. Enables audio recording and classification
     *
     * @return true if started successfully, false if permission denied
     *
     * Note: RECORD_AUDIO permission must be checked at UI layer before calling.
     */
    @SuppressLint("MissingPermission")
    fun startListening(): Boolean {
        return try {
            serviceManager.startAndBind()
            serviceManager.enableListening()
            Log.d(TAG, "Started listening via ServiceManager")
            true
        } catch (e: SecurityException) {
            Log.e(TAG, "Permission denied: ${e.message}")
            false
        }
    }

    /**
     * Stops audio monitoring completely (turns toggle OFF).
     * Also clears any active pause state.
     */
    fun stopListening() {
        serviceManager.disableListening()
        Log.d(TAG, "Stopped listening via ServiceManager")
    }

    /**
     * Temporarily pauses monitoring for the specified duration.
     * Service will auto-resume when timer expires.
     *
     * @param durationMinutes How long to pause (in minutes)
     */
    fun pauseListening(durationMinutes: Int) {
        serviceManager.pauseListening(durationMinutes)
        Log.d(TAG, "Paused for $durationMinutes minutes")
    }

    /**
     * Resumes monitoring immediately (cancels remaining pause time).
     */
    @SuppressLint("MissingPermission")
    fun resumeListening() {
        serviceManager.resumeListening()
        Log.d(TAG, "Resumed listening")
    }

    // ========================================
    // Lifecycle
    // ========================================

    override fun onCleared() {
        super.onCleared()
        // Don't unbind here - MainActivity manages service lifecycle.
        // Service should keep running even after ViewModel is destroyed.
    }
}