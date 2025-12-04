package com.example.hearmate.core.classifier

import android.content.Context
import android.util.Log
import com.example.hearmate.core.interfaces.SoundClassifier
import com.example.hearmate.core.interfaces.SoundDetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.exp

/**
 * TensorFlow Lite sound classifier using a custom-trained model.
 *
 * Model Overview:
 * ---------------
 * - Input: 15,360 audio samples (approximately 0.96 seconds at 16kHz)
 * - Output: 6 class probabilities (Alarm_Clock, Background, Car_Horn, Glass_Breaking, Gunshot, Siren)
 * - Model file: assets/vineet_model.tflite
 *
 * Processing Pipeline:
 * 1. Receive raw audio FloatArray (may need padding/trimming)
 * 2. Convert to ByteBuffer for TFLite
 * 3. Run inference to get logits
 * 4. Apply temperature scaling and softmax
 * 5. Return label with highest probability (if above threshold)
 *
 * Thread Safety:
 * - TFLite interpreter is NOT thread-safe
 * - Caller (ListeningService) must ensure single-threaded access
 * - Use Mutex in the calling code to serialize calls
 */
@Singleton
class VineetTrainedModel @Inject constructor(
    private val context: Context
) : SoundClassifier {

    companion object {
        private const val TAG = "VineetModel"

        // Model configuration
        private const val MODEL_PATH = "vineet_model.tflite"
        private const val EXPECTED_SAMPLE_SIZE = 15360  // Samples expected by model

        // Classification parameters
        private const val NOISE_CONFIDENCE_THRESHOLD = 0.80f  // Minimum confidence to report
        private const val MODEL_TEMPERATURE = 2.0f  // Softmax temperature (higher = softer distribution)

        // Emergency sound keywords for isEmergencySound() check
        private val EMERGENCY_KEYWORDS = listOf(
            "alarm", "siren", "gunshot", "glass", "breaking", "horn"
        )
    }

    // ========================================
    // Model Labels
    // ========================================

    /**
     * Class labels corresponding to model output indices.
     * Order must match the model's training labels exactly.
     */
    private val labels = listOf(
        "Alarm_Clock",     // Index 0
        "Background",      // Index 1
        "Car_Horn",        // Index 2
        "Glass_Breaking",  // Index 3
        "Gunshot",         // Index 4
        "Siren"            // Index 5
    )

    // ========================================
    // TFLite Interpreter (lazy initialization)
    // ========================================

    /**
     * TFLite interpreter instance.
     * Initialized lazily on first use to avoid blocking app startup.
     */
    private val interpreter: Interpreter by lazy {
        val model = loadModelFile()
        val interpreter = Interpreter(model)
        try {
            // Resize input tensor to expected size
            interpreter.resizeInput(0, intArrayOf(EXPECTED_SAMPLE_SIZE))
            interpreter.allocateTensors()
            Log.d(TAG, "Model initialized. Input shape: [$EXPECTED_SAMPLE_SIZE]")
        } catch (e: Exception) {
            Log.e(TAG, "Model initialization error: ${e.message}")
        }
        interpreter
    }

    // ========================================
    // SoundClassifier Interface Implementation
    // ========================================

    /**
     * Classifies an audio chunk and returns the detected sound.
     *
     * @param audioData Raw audio samples as FloatArray
     * @return SoundDetectionResult with label and confidence
     *
     * Note: This method is NOT thread-safe. Caller must synchronize access.
     */
    override suspend fun classifySound(audioData: FloatArray): SoundDetectionResult =
        withContext(Dispatchers.Default) {
            try {
                require(audioData.isNotEmpty()) { "Audio data cannot be empty" }

                // Prepare input (pad or trim to expected size)
                val processedAudio = prepareInput(audioData)

                // Convert to ByteBuffer for TFLite
                val inputBuffer = createInputBuffer(processedAudio)

                // Run inference
                val outputLogits = FloatArray(labels.size)
                interpreter.runForMultipleInputsOutputs(
                    arrayOf(inputBuffer),
                    mapOf(0 to outputLogits)
                )

                // Post-process: temperature scaling + softmax
                val probabilities = applySoftmax(outputLogits, MODEL_TEMPERATURE)

                // Find best prediction
                val maxIndex = probabilities.indexOfMax()
                var confidence = probabilities[maxIndex]
                var label = labels.getOrNull(maxIndex) ?: "Unknown"

                Log.v(TAG, "Raw Prediction: $label ($confidence)")

                // Apply confidence threshold
                if (confidence < NOISE_CONFIDENCE_THRESHOLD) {
                    label = "Noise"
                    confidence = 0.0f
                    Log.d(TAG, "Result: Noise (below threshold)")
                } else {
                    Log.d(TAG, "Result: $label ($confidence)")
                }

                SoundDetectionResult(label = label, confidence = confidence)

            } catch (e: Exception) {
                Log.e(TAG, "Classification error: ${e.message}", e)
                SoundDetectionResult(label = "Error", confidence = 0f)
            }
        }

    /**
     * Checks if the detected sound is an emergency sound.
     *
     * @param label The detected sound label
     * @return true if this is an emergency sound that should trigger an alert
     */
    override fun isEmergencySound(label: String): Boolean {
        val lowerLabel = label.lowercase()
        return EMERGENCY_KEYWORDS.any { keyword -> lowerLabel.contains(keyword) }
    }

    /**
     * Releases model resources. Call when classifier is no longer needed.
     */
    fun close() {
        try {
            interpreter.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing interpreter: ${e.message}")
        }
    }

    // ========================================
    // Internal - Input Preparation
    // ========================================

    /**
     * Prepares audio input for the model.
     * Pads with zeros if too short, trims if too long.
     */
    private fun prepareInput(audioData: FloatArray): FloatArray {
        return when {
            audioData.size < EXPECTED_SAMPLE_SIZE -> {
                // Pad with zeros at the end
                FloatArray(EXPECTED_SAMPLE_SIZE).also { padded ->
                    System.arraycopy(audioData, 0, padded, 0, audioData.size)
                }
            }
            audioData.size > EXPECTED_SAMPLE_SIZE -> {
                // Take only the first N samples
                audioData.sliceArray(0 until EXPECTED_SAMPLE_SIZE)
            }
            else -> audioData
        }
    }

    /**
     * Creates a ByteBuffer from audio data for TFLite input.
     */
    private fun createInputBuffer(audioData: FloatArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(EXPECTED_SAMPLE_SIZE * 4)  // 4 bytes per float
        buffer.order(ByteOrder.nativeOrder())
        for (value in audioData) {
            buffer.putFloat(value)
        }
        buffer.rewind()
        return buffer
    }

    // ========================================
    // Internal - Post-Processing
    // ========================================

    /**
     * Applies temperature-scaled softmax to logits.
     *
     * Temperature scaling:
     * - Temperature > 1: Softer distribution (less confident)
     * - Temperature < 1: Sharper distribution (more confident)
     * - Temperature = 1: Standard softmax
     *
     * Formula: softmax(logits / temperature)
     */
    private fun applySoftmax(logits: FloatArray, temperature: Float): FloatArray {
        val scaledLogits = logits.map { it / temperature }.toFloatArray()

        // Find max for numerical stability
        val maxLogit = scaledLogits.maxOrNull() ?: 0f

        // Calculate exp(x - max) to prevent overflow
        val expValues = FloatArray(scaledLogits.size) { i ->
            exp(scaledLogits[i] - maxLogit)
        }

        // Normalize to sum to 1
        val sum = expValues.sum()
        return FloatArray(expValues.size) { i -> expValues[i] / sum }
    }

    /**
     * Finds index of maximum value in array.
     */
    private fun FloatArray.indexOfMax(): Int {
        if (isEmpty()) return -1
        var maxIndex = 0
        var maxValue = this[0]
        for (i in 1 until size) {
            if (this[i] > maxValue) {
                maxValue = this[i]
                maxIndex = i
            }
        }
        return maxIndex
    }

    // ========================================
    // Internal - Model Loading
    // ========================================

    /**
     * Loads the TFLite model file from assets.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }
}