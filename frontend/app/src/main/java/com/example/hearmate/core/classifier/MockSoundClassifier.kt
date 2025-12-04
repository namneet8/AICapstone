package com.example.hearmate.core.classifier

import com.example.hearmate.core.interfaces.SoundClassifier
import com.example.hearmate.core.interfaces.SoundDetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Mock implementation for testing without TensorFlow model
 * Uses simple frequency analysis to simulate sound classification
 */
@Singleton
class MockSoundClassifier @Inject constructor() : SoundClassifier {

    companion object {
        // Emergency sound keywords for detection
        private val EMERGENCY_KEYWORDS = listOf(
            "siren", "alarm", "fire", "smoke", "emergency", "ambulance", "police"
        )

        // Frequency ranges (Hz) for mock classification
        private val FREQ_RANGES = mapOf(
            "Siren" to 300.0..1500.0,
            "Fire alarm" to 2000.0..4000.0,
            "Smoke alarm" to 3000.0..4500.0,
            "Car alarm" to 400.0..2000.0
        )

        private const val SAMPLE_RATE = 44100
        private const val MIN_AMPLITUDE = 0.005f
    }

    override suspend fun classifySound(audioData: FloatArray): SoundDetectionResult =
        withContext(Dispatchers.Default) {
            val amplitude = calculateRMS(audioData)

            // Silent audio -> ambient noise
            if (amplitude < MIN_AMPLITUDE) {
                return@withContext SoundDetectionResult(
                    label = "Silence",
                    confidence = 0.95f
                )
            }

            // Find dominant frequency
            val dominantFreq = findDominantFrequency(audioData)

            // Match frequency to sound type
            val (label, baseConfidence) = FREQ_RANGES.entries.find { (_, range) ->
                dominantFreq in range
            }?.let { it.key to 0.75f } ?: ("Unknown" to 0.4f)

            // Add randomness to simulate real model variance
            val confidence = (baseConfidence + Random.Default.nextFloat() * 0.2f).coerceIn(0f, 1f)

            SoundDetectionResult(label, confidence)
        }

    override fun isEmergencySound(label: String): Boolean {
        // Check if label contains any emergency keyword
        return EMERGENCY_KEYWORDS.any { keyword ->
            label.lowercase().contains(keyword)
        }
    }

    // Calculate Root Mean Square (audio volume level)
    private fun calculateRMS(audioData: FloatArray): Float {
        val sumSquares = audioData.sumOf { (it * it).toDouble() }
        return sqrt(sumSquares / audioData.size).toFloat()
    }

    // Find dominant frequency using simplified FFT
    private fun findDominantFrequency(audioData: FloatArray): Double {
        val fftSize = minOf(audioData.size, 1024)
        val fft = performSimpleFFT(audioData.take(fftSize).toFloatArray())

        // Find frequency bin with highest magnitude
        var maxMagnitude = 0.0f
        var maxIndex = 0

        for (i in 1 until fft.size / 2) {
            val magnitude = sqrt(fft[i * 2].pow(2) + fft[i * 2 + 1].pow(2))
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude
                maxIndex = i
            }
        }

        // Convert bin index to frequency (Hz)
        return (maxIndex * SAMPLE_RATE.toDouble()) / fftSize
    }

    // Simplified FFT implementation (for mock only)
    private fun performSimpleFFT(audioData: FloatArray): FloatArray {
        val size = audioData.size
        val result = FloatArray(size * 2) // Real and imaginary parts

        for (k in 0 until size) {
            var realSum = 0.0f
            var imagSum = 0.0f

            for (n in 0 until size) {
                val angle = -2.0 * PI * k * n / size
                realSum += audioData[n] * cos(angle).toFloat()
                imagSum += audioData[n] * sin(angle).toFloat()
            }

            result[k * 2] = realSum
            result[k * 2 + 1] = imagSum
        }

        return result
    }
}