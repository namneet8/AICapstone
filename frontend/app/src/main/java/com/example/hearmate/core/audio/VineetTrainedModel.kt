import android.content.Context
import com.example.hearmate.core.audio.SoundClassifier
import com.example.hearmate.core.audio.SoundDetectionResult
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

@Singleton
class VineetTrainedModel @Inject constructor(
    private val context: Context
) : SoundClassifier {

    companion object {
        private const val MODEL_PATH = "vineet_trained_model.tflite"
        private const val EXPECTED_SAMPLE_SIZE = 15360

        private val EMERGENCY_KEYWORDS = listOf(
            "alarm", "siren", "gunshot", "glass", "breaking", "horn"
        )
    }

    private val interpreter: Interpreter by lazy {
        val model = loadModelFile()
        Interpreter(model)
    }

    private val labels: List<String> by lazy { loadLabels() }

    override suspend fun classifySound(audioData: FloatArray): SoundDetectionResult =
        withContext(Dispatchers.Default) {
            try {
                require(audioData.isNotEmpty()) { "Audio data cannot be empty" }

                android.util.Log.d("YAMNet", "Audio size: ${audioData.size}")

                // Prepara input buffer (float = 4 byte)
                val inputBuffer = ByteBuffer.allocateDirect(EXPECTED_SAMPLE_SIZE * 4)
                    .order(ByteOrder.nativeOrder())

                val processedAudio = when {
                    audioData.size < EXPECTED_SAMPLE_SIZE -> {
                        audioData + FloatArray(EXPECTED_SAMPLE_SIZE - audioData.size)
                    }
                    audioData.size > EXPECTED_SAMPLE_SIZE -> {
                        audioData.sliceArray(0 until EXPECTED_SAMPLE_SIZE)
                    }
                    else -> audioData
                }

                processedAudio.forEach { inputBuffer.putFloat(it) }
                inputBuffer.rewind()

                // Prepara output: array di 1 elemento che contiene un FloatArray di dimensione 5
                val outputProbabilities = Array(1) { FloatArray(6) }
                val outputs = mapOf<Int, Any>(0 to outputProbabilities)

                // Esegui inferenza
                interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

                // Estrai i risultati dal primo (e unico) array
                val probabilities = outputProbabilities[0]
                val classId = probabilities.indexOfMax()
                val confidence = probabilities[classId]
                val labelName = labels.getOrNull(classId) ?: "Unknown"

                android.util.Log.d("YAMNet", "Detected: $labelName (confidence: $confidence)")
                android.util.Log.d("YAMNet", "Probabilities: ${probabilities.contentToString()}")

                SoundDetectionResult(
                    label = labelName,
                    confidence = confidence
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

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(): List<String> {
        return listOf(
            "Alarm_Clock",
            "Background",
            "Car_Horn",
            "Glass_Breaking",
            "Gunshot",
            "Siren"
        )
    }

    private fun FloatArray.indexOfMax(): Int {
        var maxIndex = 0
        var maxValue = this[0]
        for (i in 1 until this.size) {
            if (this[i] > maxValue) {
                maxValue = this[i]
                maxIndex = i
            }
        }
        return maxIndex
    }

    fun close() {
        try {
            interpreter.close()
        } catch (e: Exception) {
            android.util.Log.w("YAMNet", "Error closing interpreter: ${e.message}")
        }
    }
}
