package com.example.hearmate.core.audio

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