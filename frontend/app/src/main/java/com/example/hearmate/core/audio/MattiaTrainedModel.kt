import android.content.Context
import com.example.hearmate.core.audio.SoundClassifier
import com.example.hearmate.core.audio.SoundDetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.exp

@Singleton
class MattiaTrainedModel @Inject constructor(
    private val context: Context
) : SoundClassifier {

    companion object {
        private const val MODEL_PATH = "mattia_trained_model.tflite"
        private const val NOISE_CONFIDENCE_THRESHOLD = 0.75f  // ✅ Threshold per "Noise"

        private val EMERGENCY_KEYWORDS = listOf(
            "alarm", "siren", "gunshot", "glass", "breaking", "horn"
        )
    }

    private val interpreter: Interpreter by lazy {
        val model = loadModelFile()
        val options = Interpreter.Options().apply {
            setNumThreads(4)  // Usa 4 thread per performance migliori
        }
        Interpreter(model, options)
    }

    private val labels: List<String> by lazy { loadLabels() }

    override suspend fun classifySound(audioData: FloatArray): SoundDetectionResult =
        withContext(Dispatchers.Default) {
            try {
                require(audioData.isNotEmpty()) { "Audio data cannot be empty" }

                android.util.Log.d("YAMNet", "Audio size: ${audioData.size}")

                // ========================================
                // STEP 1: NORMALIZZA AUDIO (se necessario)
                // ========================================
                // Se audioData è già normalizzato (-1 a 1), salta questo step
                // Se viene da Short[], allora è già normalizzato nel caller
                val normalizedAudio = audioData

                // ========================================
                // STEP 2: RESIZE INPUT TENSOR (CRITICO!)
                // ========================================
                // Questo permette di accettare audio di qualsiasi lunghezza
                interpreter.resizeInput(0, intArrayOf(normalizedAudio.size))
                interpreter.allocateTensors()

                // Debug: verifica shape
                val inputTensor = interpreter.getInputTensor(0)
                android.util.Log.d("YAMNet", "Input tensor shape after resize: ${inputTensor.shape().contentToString()}")

                // ========================================
                // STEP 3: PREPARA OUTPUT
                // ========================================
                // Output è un array con probabilità per ogni classe
                val outputProbabilities = FloatArray(labels.size)  // ✅ Cambiato da Array(1) { FloatArray() }

                // ========================================
                // STEP 4: RUN INFERENCE
                // ========================================
                // Usa run() semplice invece di runForMultipleInputsOutputs
                interpreter.run(normalizedAudio, outputProbabilities)

                // ========================================
                // STEP 5: PROCESSA RISULTATI
                // ========================================
                // Applica softmax per convertire logits in probabilità
                val probabilities = softmax(outputProbabilities)

                val classId = probabilities.indexOfMax()
                val confidence = probabilities[classId]

                // ✅ AGGIUNGI: Se confidence < threshold, predici "Noise"
                val labelName = if (confidence < NOISE_CONFIDENCE_THRESHOLD) {
                    "Noise"
                } else {
                    labels.getOrNull(classId) ?: "Unknown"
                }

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

    /**
     * Applica la funzione softmax per convertire logits in probabilità.
     * Questo assicura che le probabilità sommino a 1.0
     */
    private fun softmax(logits: FloatArray): FloatArray {
        // Sottrai il massimo per stabilità numerica
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()

        // Se somma è 0, ritorna uniforme
        if (sumExps == 0f || sumExps.isNaN()) {
            return FloatArray(logits.size) { 1f / logits.size }
        }

        return exps.map { it / sumExps }.toFloatArray()
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
        // Aggiorna con le tue classi esatte da class_names.txt
        return listOf(
            "Alarm Clock",
            "Background",
            "Siren",
            "Speech",
            "Vehicle Horn"
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