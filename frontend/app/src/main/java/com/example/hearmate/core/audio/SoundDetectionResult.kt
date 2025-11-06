package com.example.hearmate.core.audio

/**
 * Result of sound classification containing label and confidence score
 */
data class SoundDetectionResult(
    val label: String,              // Sound label (e.g., "Siren", "Fire alarm")
    val confidence: Float,          // Confidence score (0.0 to 1.0)
    val timestamp: Long = System.currentTimeMillis()
)