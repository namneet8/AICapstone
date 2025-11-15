package com.example.hearmate.presentation.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccessTimeFilled
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.hearmate.presentation.viewmodel.ListeningViewModel
import com.example.hearmate.core.audio.SoundDetectionResult
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState

import kotlinx.coroutines.delay


private fun formatTimestamp(timestamp: Long): String {
    val dateFormat = java.text.SimpleDateFormat("MMM dd, HH:mm:ss", java.util.Locale.getDefault())
    return dateFormat.format(java.util.Date(timestamp))
}

@OptIn(ExperimentalMaterial3Api::class)
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
    val emergencySound by viewModel.emergencySound.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("HearMate") },
                actions = {
                    IconButton(onClick = onSettingsClick) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Settings",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Main content
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // Status indicator with modern design
                ModernStatusIndicator(
                    isListening = isRecording,
                    isRecording = isCapturing
                )

                Spacer(modifier = Modifier.height(48.dp))

                // Last detected sound card
                lastSound?.let {
                    ModernSoundInfoCard(it)
                    Spacer(modifier = Modifier.height(32.dp))
                }

                // Modern control buttons
                Row(
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    FilledTonalButton(
                        onClick = { viewModel.startListening() },
                        enabled = !isRecording,
                        modifier = Modifier
                            .weight(1f)
                            .height(56.dp),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Text(
                            "Start",
                            style = MaterialTheme.typography.titleMedium
                        )
                    }

                    FilledTonalButton(
                        onClick = { viewModel.stopListening() },
                        enabled = isRecording,
                        modifier = Modifier
                            .weight(1f)
                            .height(56.dp),
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.filledTonalButtonColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer,
                            contentColor = MaterialTheme.colorScheme.onErrorContainer
                        )
                    ) {
                        Text(
                            "Stop",
                            style = MaterialTheme.typography.titleMedium
                        )
                    }
                }
            }

            // Emergency alert overlay with auto-dismiss
            if (isEmergencyAlert) {
                EmergencyAlertOverlay(
                    detectedSound = emergencySound,
                    onAutoDismiss = { viewModel.dismissEmergencyAlert() }
                )
            }
        }
    }
}

@Composable
private fun ModernStatusIndicator(
    isListening: Boolean,
    isRecording: Boolean
) {
    val statusText = when {
        !isListening -> "Paused"
        isRecording -> "Recording Sound"
        else -> "Monitoring"
    }

    val statusColor = when {
        !isListening -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        isRecording -> Color(0xFFE53935)
        else -> Color(0xFF43A047)
    }

    // Pulsing animation for the indicator
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Large circular indicator
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(160.dp)
                .shadow(
                    elevation = if (isListening) 12.dp else 4.dp,
                    shape = CircleShape,
                    ambientColor = statusColor.copy(alpha = 0.3f),
                    spotColor = statusColor.copy(alpha = 0.3f)
                )
                .clip(CircleShape)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            statusColor.copy(alpha = 0.3f),
                            statusColor.copy(alpha = 0.1f)
                        )
                    )
                )
        ) {
            Box(
                modifier = Modifier
                    .size(if (isListening && isRecording) 80.dp * scale else 80.dp)
                    .clip(CircleShape)
                    .background(statusColor)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = statusText,
            style = MaterialTheme.typography.headlineMedium,
            color = statusColor
        )
    }
}

@Composable
private fun ModernSoundInfoCard(sound: SoundDetectionResult) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(
                elevation = 4.dp,
                shape = RoundedCornerShape(20.dp)
            ),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = sound.label,
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )

                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.2f)
                ) {
                    Text(
                        text = "${"%.0f".format(sound.confidence * 100)}%",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }

            HorizontalDivider(
                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.2f)
            )

            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.AccessTimeFilled,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.6f),
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = formatTimestamp(sound.timestamp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                )
            }
        }
    }
}

@Composable
private fun EmergencyAlertOverlay(
    detectedSound: SoundDetectionResult?,
    onAutoDismiss: () -> Unit
) {
    // Subtle pulse animation
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale_pulse"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.85f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .padding(32.dp)
                .fillMaxWidth()
                .shadow(24.dp, RoundedCornerShape(32.dp)),
            shape = RoundedCornerShape(32.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color.White
            )
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier.padding(40.dp)
            ) {
                // Warning icon with pulse animation
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.size(120.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(120.dp * scale)
                            .clip(CircleShape)
                            .background(Color(0xFFFFEBEE)),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "⚠️",
                            style = MaterialTheme.typography.displayLarge,
                        )
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                Text(
                    text = "Emergency Detected",
                    style = MaterialTheme.typography.headlineMedium,
                    color = Color(0xFF212121)
                )

                Spacer(modifier = Modifier.height(12.dp))

                detectedSound?.let {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = Color(0xFFE53935).copy(alpha = 0.1f),
                        modifier = Modifier.padding(vertical = 8.dp)
                    ) {
                        Text(
                            text = it.label,
                            style = MaterialTheme.typography.titleLarge,
                            color = Color(0xFFE53935),
                            modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Stay alert and check your surroundings",
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color(0xFF757575)
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Dismiss button
                FilledTonalButton(
                    onClick = onAutoDismiss,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.filledTonalButtonColors(
                        containerColor = Color(0xFFE53935),
                        contentColor = Color.White
                    )
                ) {
                    Text(
                        "Dismiss",
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }
        }
    }
}