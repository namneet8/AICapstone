package com.example.hearmate.presentation.ui.activities

import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.hearmate.presentation.ui.theme.HearMateTheme

/**
 * Fullscreen emergency alert Activity.
 *
 * Purpose:
 * --------
 * Displays a prominent, color-coded alert when an emergency sound is detected.
 * Designed to grab the user's attention even if the phone was locked or in pocket.
 *
 * Features:
 * - Wakes the screen and shows over lock screen
 * - Color-coded background based on sound type
 * - Auto-dismisses after 10 seconds
 * - Manual dismiss button
 *
 * Launch Methods:
 * - Foreground: Started directly by ListeningService
 * - Background: Started via notification with fullScreenIntent
 */
class EmergencyAlertActivity : ComponentActivity() {

    companion object {
        /** Intent extra key for the detected sound label */
        const val EXTRA_SOUND_LABEL = "extra_sound_label"

        /** Auto-dismiss delay in milliseconds */
        private const val AUTO_DISMISS_DELAY_MS = 10_000L
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Configure window to show over lock screen
        setupLockScreenFlags()

        // Get sound label from intent
        val soundLabel = intent.getStringExtra(EXTRA_SOUND_LABEL) ?: "Unknown"

        // Set up the UI
        setContent {
            HearMateTheme {
                EmergencyAlertScreen(
                    soundLabel = soundLabel,
                    onDismiss = { finish() }
                )
            }
        }

        // Auto-dismiss after delay
        window.decorView.postDelayed({
            finish()
        }, AUTO_DISMISS_DELAY_MS)
    }

    /**
     * Configures window flags to show this Activity over the lock screen
     * and wake the screen when launched.
     */
    @Suppress("DEPRECATION")
    private fun setupLockScreenFlags() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            // Modern API (Android 8.1+)
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            // Legacy API
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }

        // Keep screen on while alert is visible
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
}

// ============================================
// Composable UI
// ============================================

/**
 * Fullscreen alert UI with color-coded background.
 *
 * @param soundLabel The detected emergency sound type
 * @param onDismiss Callback when user dismisses the alert
 */
@Composable
private fun EmergencyAlertScreen(
    soundLabel: String,
    onDismiss: () -> Unit
) {
    val backgroundColor = getColorForSound(soundLabel)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundColor),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier.padding(32.dp)
        ) {
            // Alert message
            Text(
                text = "HearMate detected $soundLabel sound",
                style = MaterialTheme.typography.headlineLarge,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Dismiss button
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Black
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Text("Dismiss", style = MaterialTheme.typography.titleMedium)
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Auto-dismiss hint
            Text(
                text = "Auto-dismiss in 10 seconds",
                color = Color.White.copy(alpha = 0.8f),
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

/**
 * Returns a background color based on the detected sound type.
 *
 * Color assignments:
 * - Siren: Red (danger/emergency)
 * - Car Horn: Yellow (caution/traffic)
 * - Alarm Clock: Green (routine alert)
 * - Glass Breaking: Orange (potential break-in)
 * - Gunshot: Purple (severe danger)
 * - Unknown: Dark gray (fallback)
 */
@Composable
private fun getColorForSound(label: String): Color {
    return when (label) {
        "Siren" -> Color(0xFFD32F2F)           // Red
        "Car_Horn" -> Color(0xFFFFEB3B)        // Yellow
        "Alarm_Clock" -> Color(0xFF4CAF50)     // Green
        "Glass_Breaking" -> Color(0xFFFF9800)  // Orange
        "Gunshot" -> Color(0xFF9C27B0)         // Purple
        else -> Color(0xFF212121)              // Dark gray (fallback)
    }
}