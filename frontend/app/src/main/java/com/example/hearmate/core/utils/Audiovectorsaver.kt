package com.example.hearmate.core.utils

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * utility to save audio vectors for temporary verification
 */
@Singleton
class AudioVectorSaver @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS", Locale.US)

    private val vectorFolder: File by lazy {
        File(context.getExternalFilesDir(null), "audio_vectors").apply {
            if (!exists()) mkdirs()
        }
    }

    /**
     * Save the audio vector to a text file
     */
    suspend fun saveVector(
        audioVector: FloatArray,
        predictedLabel: String,
        confidence: Float
    ) = withContext(Dispatchers.IO) {
        try {
            val timestamp = dateFormat.format(Date())
            val filename = "${timestamp}_${predictedLabel}_${String.format(Locale.US, "%.2f", confidence)}.txt"
            val file = File(vectorFolder, filename)

            // Save vector as space-separated values
            file.writeText(audioVector.joinToString(separator = " "))

            Log.d("VectorSaver", "Saved: $filename (${audioVector.size} samples)")

        } catch (e: Exception) {
            Log.e("VectorSaver", "Error: ${e.message}")
        }
    }
}