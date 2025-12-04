package com.example.hearmate.core.service

import android.Manifest
import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Manages ListeningService lifecycle and provides a clean interface for ViewModels.
 *
 * Architecture Overview:
 * ----------------------
 * This class acts as a bridge between the UI layer (ViewModel) and the Service layer.
 * It handles:
 * - Starting/stopping the foreground service
 * - Binding/unbinding to the service
 * - Proxying service states to the UI via StateFlows
 * - Queuing actions if service is not yet bound
 *
 * Why this pattern?
 * - ViewModels shouldn't directly interact with Services
 * - Service binding is asynchronous (takes time to connect)
 * - We need to queue actions if service isn't bound yet
 * - Provides a single source of truth for service state
 *
 * Usage:
 * - ViewModel calls ServiceManager methods (startAndBind, enableListening, etc.)
 * - ServiceManager proxies to the actual service
 * - UI observes StateFlows from ServiceManager
 */
@RequiresApi(Build.VERSION_CODES.O)
class ServiceManager(private val context: Context) {

    companion object {
        private const val TAG = "ServiceManager"
    }

    // ========================================
    // Service Connection State
    // ========================================

    private var service: ListeningService? = null
    private var isBound = false

    // Coroutine scope for observing service state flows
    private val managerScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val observeJobs = mutableListOf<Job>()

    // Action to execute once service is bound (if called before binding completes)
    private var pendingAction: (() -> Unit)? = null

    // ========================================
    // Exposed State Flows (proxied from Service)
    // ========================================

    /** Whether the service is currently bound and accessible */
    private val _isServiceBound = MutableStateFlow(false)
    val isServiceBound: StateFlow<Boolean> = _isServiceBound.asStateFlow()

    /** Whether audio monitoring is actively running */
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    /** Whether user has enabled listening (toggle ON state) */
    private val _isListeningEnabled = MutableStateFlow(false)
    val isListeningEnabled: StateFlow<Boolean> = _isListeningEnabled.asStateFlow()

    /** Whether monitoring is currently paused */
    private val _isPaused = MutableStateFlow(false)
    val isPaused: StateFlow<Boolean> = _isPaused.asStateFlow()

    /** Seconds remaining in pause timer */
    private val _pauseTimeRemaining = MutableStateFlow(0L)
    val pauseTimeRemaining: StateFlow<Long> = _pauseTimeRemaining.asStateFlow()

    // ========================================
    // Service Connection
    // ========================================

    /**
     * ServiceConnection callback handles bind/unbind events.
     */
    private val connection = object : ServiceConnection {

        @SuppressLint("MissingPermission")
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? ListeningService.LocalBinder
            service = localBinder?.getService()
            isBound = true
            _isServiceBound.value = true

            Log.d(TAG, "Service connected")

            // Start observing all service state flows
            observeServiceStates()

            // Execute any action that was queued before binding completed
            pendingAction?.invoke()
            pendingAction = null
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            isBound = false
            _isServiceBound.value = false
            resetLocalStates()
            cancelObservation()

            Log.d(TAG, "Service disconnected")
        }
    }

    /**
     * Resets all local state flows when service disconnects.
     */
    private fun resetLocalStates() {
        _isListening.value = false
        _isListeningEnabled.value = false
        _isPaused.value = false
        _pauseTimeRemaining.value = 0L
    }

    /**
     * Starts collecting service state flows and mirrors them to local flows.
     * This keeps UI in sync with service state.
     */
    private fun observeServiceStates() {
        val svc = service ?: return
        cancelObservation()

        // Mirror each service flow to our local flow
        observeJobs.add(managerScope.launch {
            svc.isListening.collect { _isListening.value = it }
        })

        observeJobs.add(managerScope.launch {
            svc.isListeningEnabled.collect { _isListeningEnabled.value = it }
        })

        observeJobs.add(managerScope.launch {
            svc.isPaused.collect { _isPaused.value = it }
        })

        observeJobs.add(managerScope.launch {
            svc.pauseTimeRemaining.collect { _pauseTimeRemaining.value = it }
        })
    }

    /**
     * Cancels all state observation jobs.
     */
    private fun cancelObservation() {
        observeJobs.forEach { it.cancel() }
        observeJobs.clear()
    }

    // ========================================
    // Public API - Service Lifecycle
    // ========================================

    /**
     * Starts the foreground service and binds to it.
     * Call this when user enables listening.
     */
    fun startAndBind() {
        val intent = Intent(context, ListeningService::class.java)
        context.startForegroundService(intent)
        bind()
        Log.d(TAG, "Service started and binding initiated")
    }

    /**
     * Binds to an already running service.
     * Use this to reconnect when Activity comes to foreground.
     */
    fun bind() {
        if (!isBound) {
            val intent = Intent(context, ListeningService::class.java)
            context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }
    }

    /**
     * Unbinds from service without stopping it.
     * Service continues running in background.
     */
    fun unbind() {
        if (isBound) {
            cancelObservation()
            context.unbindService(connection)
            isBound = false
            _isServiceBound.value = false
            service = null
            Log.d(TAG, "Service unbound")
        }
    }

    /**
     * Stops the service completely.
     * Use this only when you want to fully stop the service.
     */
    fun stopService() {
        service?.disableListening()
        val intent = Intent(context, ListeningService::class.java)
        context.stopService(intent)
        Log.d(TAG, "Service stopped")
    }

    // ========================================
    // Public API - Listening Control
    // ========================================

    /**
     * Enables audio monitoring (turns toggle ON).
     * If service not yet bound, queues the action for later execution.
     *
     * @return true if action was executed or queued successfully
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun enableListening(): Boolean {
        return if (isBound && service != null) {
            service?.enableListening() ?: false
        } else {
            // Service not ready yet - queue action for when it connects
            pendingAction = { service?.enableListening() }
            Log.d(TAG, "enableListening queued - waiting for service bind")
            true
        }
    }

    /**
     * Disables audio monitoring (turns toggle OFF).
     */
    fun disableListening() {
        if (isBound && service != null) {
            service?.disableListening()
        } else {
            pendingAction = null
        }
    }

    /**
     * Pauses monitoring for specified duration.
     *
     * @param durationMinutes Pause duration in minutes
     */
    fun pauseListening(durationMinutes: Int) {
        if (isBound && service != null) {
            service?.pauseListening(durationMinutes)
        } else {
            Log.w(TAG, "Cannot pause - service not bound")
        }
    }

    /**
     * Resumes monitoring immediately.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun resumeListening() {
        if (isBound && service != null) {
            service?.resumeListening()
        } else {
            Log.w(TAG, "Cannot resume - service not bound")
        }
    }

    // ========================================
    // Cleanup
    // ========================================

    /**
     * Cleans up resources. Call when ServiceManager is no longer needed.
     */
    fun destroy() {
        unbind()
        managerScope.cancel()
    }
}