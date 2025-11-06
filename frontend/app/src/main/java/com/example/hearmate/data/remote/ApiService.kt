package com.example.hearmate.data.remote

import com.example.hearmate.data.remote.model.SoundEventUploadRequest
import com.example.hearmate.data.remote.model.UploadResponse
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

/**
 * Retrofit API service for MongoDB communication
 * Defines endpoints for uploading sound events
 */
interface ApiService {

    /**
     * Upload multiple sound events to MongoDB
     * POST /api/sound-events/batch
     *
     * @param request Contains list of sound events to upload
     * @return Response with upload status
     */
    @POST("api/sound-events/batch")
    suspend fun uploadEvents(
        @Body request: SoundEventUploadRequest
    ): Response<UploadResponse>

    /**
     * Health check endpoint to verify API connectivity
     * GET /api/health
     */
    @GET("api/health")
    suspend fun healthCheck(): Response<Map<String, String>>
}