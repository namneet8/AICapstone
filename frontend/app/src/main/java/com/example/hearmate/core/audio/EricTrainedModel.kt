import android.content.Context
import com.example.hearmate.core.audio.SoundClassifier
import com.example.hearmate.core.audio.SoundDetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import javax.inject.Inject
import javax.inject.Singleton


@Singleton
class EricTrainedModel @Inject constructor(
    private val context: Context
) : SoundClassifier {

    companion object {
        private const val MODEL_PATH = "eric_trained_model.tflite"

        // Keywords indicating emergency sounds in YAMNet's 521 classes
        private val EMERGENCY_KEYWORDS = listOf(
            "siren", "alarm", "fire", "smoke", "emergency",
            "ambulance", "police", "civil defense", "vehicle horn"
        )
    }

    // Lazy initialization - created only when first used
    private val classifier: AudioClassifier by lazy {
        AudioClassifier.createFromFile(context, MODEL_PATH)
    }

    // TensorAudio buffer for model input
    private val tensorAudio by lazy {
        classifier.createInputTensorAudio()
    }

    // Load YAMNet labels (521 lines)
    private val labels: List<String> by lazy { loadLabels() }

    override suspend fun classifySound(audioData: FloatArray): SoundDetectionResult =
        withContext(Dispatchers.Default) {
            try {
                require(audioData.isNotEmpty()) { "Audio data cannot be empty" }

                // Debug logging
                android.util.Log.d("YAMNet", "Audio size: ${audioData.size}, Sample rate expected: 16kHz")
                android.util.Log.d("YAMNet", "Audio range: ${audioData.minOrNull()} to ${audioData.maxOrNull()}")

                // Load audio into tensor
                tensorAudio.load(audioData)

                // Run classification
                val results = classifier.classify(tensorAudio)

                android.util.Log.d("YAMNet", "Results size: ${results.size}")

                // Get top result (highest confidence)
                val topCategory = results
                    .firstOrNull()?.categories
                    ?.maxByOrNull { it.score }

                // Get label from CSV (display_name column)
                val labelName = labels.getOrNull(topCategory?.index ?: -1) ?: "Unknown"

                android.util.Log.d("YAMNet", "Top: $labelName (${topCategory?.score})")

                SoundDetectionResult(
                    label = labelName,
                    confidence = topCategory?.score ?: 0f
                )

            } catch (e: Exception) {
                android.util.Log.e("YAMNet", "Classification error: ${e.message}", e)
                e.printStackTrace()
                SoundDetectionResult(label = "Error: ${e.message}", confidence = 0f)
            }
        }

    override fun isEmergencySound(label: String): Boolean {
        val lowerLabel = label.lowercase()
        return EMERGENCY_KEYWORDS.any { keyword ->
            lowerLabel.contains(keyword)
        }
    }


    private fun loadLabels(): List<String> {
        return context.assets.open("eric_trained_map.csv").bufferedReader().useLines { lines ->
            lines.drop(1) // skip header
                .map { line ->
                    // The third column is display_name
                    val parts = line.split(",")
                    if (parts.size >= 3) parts[2].replace("\"", "") else "Unknown"
                }
                .toList()
        }
    }
}
