package com.example.hearmate.core.audio

/**
 * Result of sound classification containing label and confidence score
 */
data class SoundDetectionResult(
    val label: String,              // Sound label (e.g., "Siren", "Fire alarm")
    val confidence: Float,          // Confidence score (0.0 to 1.0)
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Interface for sound classification implementations
 * Allows switching between Mock and Real (YAMNet) classifiers
 */
interface SoundClassifier {
    /**
     * Classify audio data and return detection result
     * @param audioData Raw audio samples as FloatArray
     * @return SoundDetectionResult with label and confidence
     */
    suspend fun classifySound(audioData: FloatArray): SoundDetectionResult

    /**
     * Check if detected sound is an emergency sound
     * @param label The detected sound label
     * @return true if emergency sound, false otherwise
     */
    fun isEmergencySound(label: String): Boolean
}