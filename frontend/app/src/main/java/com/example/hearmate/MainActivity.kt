package com.example.hearmate

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.core.net.toUri
import androidx.navigation.compose.rememberNavController
import com.example.hearmate.core.service.ServiceManager
import com.example.hearmate.navigation.AppNavHost
import com.example.hearmate.presentation.ui.theme.HearMateTheme
import com.example.hearmate.utils.PermissionUtils
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * Main entry point Activity for the HearMate app.
 *
 * Responsibilities:
 * -----------------
 * 1. Set up the Compose UI with navigation
 * 2. Request necessary permissions (microphone, notifications)
 * 3. Manage service binding lifecycle
 *
 * Service Binding Strategy:
 * - onCreate: Initial bind to sync state
 * - onStart: Re-bind when becoming visible
 * - onStop: Unbind (but don't stop service)
 *
 * The service continues running independently even when Activity is destroyed.
 * This allows continuous monitoring in the background.
 */
@RequiresApi(Build.VERSION_CODES.O)
@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject
    lateinit var serviceManager: ServiceManager

    /**
     * Permission launcher for microphone access.
     * Shows toast feedback based on result.
     */
    private val microphonePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(
                this,
                "Microphone permission is required for sound detection",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    // ========================================
    // Activity Lifecycle
    // ========================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Request permissions if not already granted
        requestPermissionsIfNeeded()

        // Check full screen intent permission (Android 14+)
        checkFullScreenIntentPermission()

        // Bind to service to sync current state
        // Service may already be running with preserved pause state
        serviceManager.bind()

        // Set up Compose UI
        setContent {
            HearMateTheme {
                val navController = rememberNavController()
                AppNavHost(navController)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        // Re-bind when activity becomes visible to sync state
        serviceManager.bind()
    }

    override fun onStop() {
        super.onStop()
        // Unbind but keep service running in background
        serviceManager.unbind()
    }

    // ========================================
    // Permissions
    // ========================================

    /**
     * Requests microphone permission if not already granted.
     * Called on app startup.
     */
    private fun requestPermissionsIfNeeded() {
        if (!PermissionUtils.hasRecordAudioPermission(this)) {
            microphonePermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
        }
    }

    /**
     * Checks and requests full screen intent permission (Android 14+).
     *
     * Full screen intents allow emergency alerts to wake the screen and
     * show over the lock screen. Required for proper emergency notification
     * behavior on Android 14 and later.
     */
    @SuppressLint("InlinedApi")
    private fun checkFullScreenIntentPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            val notificationManager = getSystemService(android.app.NotificationManager::class.java)

            if (!notificationManager.canUseFullScreenIntent()) {
                // Open system settings for full screen intent permission
                try {
                    val intent = Intent(
                        Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT,
                        "package:$packageName".toUri()
                    )
                    startActivity(intent)
                    Toast.makeText(
                        this,
                        "Please enable 'Full screen notifications' for emergency alerts",
                        Toast.LENGTH_LONG
                    ).show()
                } catch (e: Exception) {
                    // Fallback to general app notification settings
                    val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                        putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
                    }
                    startActivity(intent)
                }
            }
        }
    }
}