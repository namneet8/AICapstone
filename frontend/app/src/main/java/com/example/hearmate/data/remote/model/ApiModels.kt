package com.example.hearmate.data.remote.model

import com.google.gson.annotations.SerializedName

/**
 * Data Transfer Object for sound events
 * Used for API communication with MongoDB
 */
data class SoundEventDto(
    @SerializedName("label")
    val label: String,

    @SerializedName("confidence")
    val confidence: Float,

    @SerializedName("timestamp")
    val timestamp: Long,

    @SerializedName("isEmergency")
    val isEmergency: Boolean,

    @SerializedName("rmsLevel")
    val rmsLevel: Double,

    @SerializedName("localId")
    val localId: Long? = null  // Optional local ID for tracking
)

/**
 * Request model for uploading sound events to MongoDB
 */
data class SoundEventUploadRequest(
    @SerializedName("events")
    val events: List<SoundEventDto>
)

/**
 * Response from MongoDB after uploading events
 */
data class UploadResponse(
    @SerializedName("success")
    val success: Boolean,

    @SerializedName("message")
    val message: String? = null,

    @SerializedName("uploadedCount")
    val uploadedCount: Int = 0,

    @SerializedName("failedCount")
    val failedCount: Int = 0
)

/**
 * Generic API response wrapper
 */
data class ApiResponse<T>(
    @SerializedName("success")
    val success: Boolean,

    @SerializedName("data")
    val data: T? = null,

    @SerializedName("error")
    val error: String? = null
)