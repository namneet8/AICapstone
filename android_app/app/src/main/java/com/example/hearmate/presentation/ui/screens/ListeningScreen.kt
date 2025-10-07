package com.example.hearmate.presentation.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.hearmate.core.audio.SoundDetectionResult
import com.example.hearmate.presentation.viewmodel.ListeningViewModel


private fun formatTimestamp(timestamp: Long): String {
    val dateFormat = java.text.SimpleDateFormat("MMM dd, HH:mm:ss", java.util.Locale.getDefault())
    return dateFormat.format(java.util.Date(timestamp))
}
@Composable
fun ListeningScreen(
    viewModel: ListeningViewModel,
    onSettingsClick: () -> Unit
) {
    // Collect states from ViewModel
    val lastSound by viewModel.lastDetectedSound.collectAsState()
    val isRecording by viewModel.isListening.collectAsState()
    val isCapturing by viewModel.isThresholdTriggered.collectAsState()
    val isEmergencyAlert by viewModel.isEmergencyAlert.collectAsState()

    Box(modifier = Modifier.fillMaxSize()) {
        // Main content
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Status indicator
            StatusIndicator(
                isListening = isRecording,
                isRecording = isCapturing
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Last detected sound card
            lastSound?.let {
                SoundInfoCard(it)
                Spacer(modifier = Modifier.height(24.dp))
            }

            // Control buttons
            Row {
                Button(
                    onClick = { viewModel.startListening() },
                    enabled = !isRecording
                ) {
                    Text("Start")
                }

                Spacer(modifier = Modifier.width(16.dp))

                Button(
                    onClick = { viewModel.stopListening() },
                    enabled = isRecording
                ) {
                    Text("Stop")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = onSettingsClick) {
                Text("Settings")
            }
        }

        // Emergency alert overlay
        if (isEmergencyAlert) {
            EmergencyAlertOverlay(
                onDismiss = { viewModel.dismissEmergencyAlert() }
            )
        }
    }
}

@Composable
private fun StatusIndicator(
    isListening: Boolean,
    isRecording: Boolean
) {
    // Determine status text and color
    val statusText = when {
        !isListening -> "Not Listening"
        isRecording -> "Recording" // Above threshold, capturing 2 seconds
        else -> "Listening" // Below threshold, monitoring
    }

    val statusColor = when {
        !isListening -> MaterialTheme.colorScheme.onSurface
        isRecording -> Color(0xFFE53935) // Red = recording
        else -> Color(0xFF43A047) // Green = listening/monitoring
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        // Colored dot indicator
        if (isListening) {
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .clip(CircleShape)
                    .background(statusColor)
            )
            Spacer(modifier = Modifier.width(8.dp))
        }

        // Status text
        Text(
            text = statusText,
            style = MaterialTheme.typography.headlineMedium,
            color = statusColor
        )
    }
}

@Composable
private fun SoundInfoCard(sound: SoundDetectionResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Detected: ${sound.label}",
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = "Confidence: ${"%.1f".format(sound.confidence * 100)}%",
                style = MaterialTheme.typography.bodyMedium
            )

            Text(
                text = "At Time: ${formatTimestamp(sound.timestamp)}",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
private fun EmergencyAlertOverlay(
    onDismiss: () -> Unit
) {
    // Infinite flashing animation
    val infiniteTransition = rememberInfiniteTransition(label = "emergency_flash")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.9f,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha_flash"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Red.copy(alpha = alpha)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "🚨",
                style = MaterialTheme.typography.displayLarge,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "EMERGENCY SOUND DETECTED",
                style = MaterialTheme.typography.headlineMedium,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Red
                )
            ) {
                Text("DISMISS")
            }
        }
    }

}