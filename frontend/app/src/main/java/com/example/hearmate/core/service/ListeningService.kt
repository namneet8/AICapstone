package com.example.hearmate.core.service

import android.Manifest
import android.R
import com.example.hearmate.core.audio.SoundDetectionResult
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.annotation.RequiresPermission
import androidx.core.app.NotificationCompat
import com.example.hearmate.core.audio.AudioRecorderManager
import com.example.hearmate.core.audio.SoundClassifier
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

@AndroidEntryPoint
class ListeningService : Service() {

    @Inject
    lateinit var audioRecorderManager: AudioRecorderManager

    @Inject
    lateinit var soundClassifier: SoundClassifier

    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _lastDetectedSound = MutableStateFlow<SoundDetectionResult?>(null)
    val lastDetectedSound: StateFlow<SoundDetectionResult?> = _lastDetectedSound.asStateFlow()

    override fun onBind(intent: Intent): IBinder = LocalBinder()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        startForeground(1, createNotification())
        return START_STICKY
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startListening(): Boolean {
        return if (audioRecorderManager.startRecording()) {
            _isListening.value = true
            true
        } else false
    }

    fun stopListening() {
        audioRecorderManager.stopRecording()
        _isListening.value = false
        stopSelf()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "listening_channel",
                "Audio Listening",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, "listening_channel")
            .setContentTitle("HearMate")
            .setContentText("Listening for sounds...")
            .setSmallIcon(R.drawable.ic_media_play)
            .build()
    }

    override fun onDestroy() {
        audioRecorderManager.onDestroy()
        super.onDestroy()
    }

    inner class LocalBinder : Binder() {
        fun getService(): ListeningService = this@ListeningService
    }
}