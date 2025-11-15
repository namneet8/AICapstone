package com.example.hearmate.data.repository

import android.util.Log
import com.example.hearmate.data.local.dao.SoundEventDao
import com.example.hearmate.data.local.entity.SoundEventEntity
import com.example.hearmate.core.audio.SoundDetectionResult
import com.example.hearmate.data.remote.ApiService
import com.example.hearmate.data.remote.model.SoundEventDto
import com.example.hearmate.data.remote.model.SoundEventUploadRequest
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing sound event data
 * Handles both local database (Room) and remote sync (MongoDB)
 */
@Singleton
class SoundEventRepository @Inject constructor(
    private val soundEventDao: SoundEventDao,
    private val apiService: ApiService
) {

    private val TAG = "SoundEventRepository"

    // ========== LOCAL DATABASE OPERATIONS ==========

    /**
     * Save a new sound detection event to local database
     */
    suspend fun saveEvent(
        result: SoundDetectionResult,
        isEmergency: Boolean,
        rmsLevel: Double
    ): Long {
        val entity = SoundEventEntity(
            label = result.label,
            confidence = result.confidence,
            timestamp = result.timestamp,
            isEmergency = isEmergency,
            rmsLevel = rmsLevel,
            isSynced = false
        )
        return soundEventDao.insertEvent(entity)
    }

    /**
     * Get all events as Flow (observes changes)
     */
    fun getAllEvents(): Flow<List<SoundEventEntity>> {
        return soundEventDao.getAllEvents()
    }

    /**
     * Get only emergency events
     */
    fun getEmergencyEvents(): Flow<List<SoundEventEntity>> {
        return soundEventDao.getEmergencyEvents()
    }

    /**
     * Get events within time range
     */
    suspend fun getEventsByTimeRange(startTime: Long, endTime: Long): List<SoundEventEntity> {
        return soundEventDao.getEventsByTimeRange(startTime, endTime)
    }

    /**
     * Get total event count
     */
    suspend fun getEventCount(): Int {
        return soundEventDao.getEventCount()
    }

    /**
     * Get count of unsynced events
     */
    suspend fun getUnsyncedCount(): Int {
        return soundEventDao.getUnsyncedEvents().size
    }

    // ========== SYNC OPERATIONS ==========

    /**
     * Sync unsynced events to MongoDB
     * @return Pair of (successCount, failureCount). Returns (-1, -1) on network error
     */
    suspend fun syncToMongoDB(): Pair<Int, Int> = try {
        val unsyncedEvents = soundEventDao.getUnsyncedEvents()

        if (unsyncedEvents.isEmpty()) {
            Log.d(TAG, "No events to sync")
            Pair(0, 0)  // ✅ ora non c’è return interno
        } else {
            Log.d(TAG, "Syncing ${unsyncedEvents.size} events to MongoDB")

            val dtos = unsyncedEvents.map { entity ->
                SoundEventDto(
                    label = entity.label,
                    confidence = entity.confidence,
                    timestamp = entity.timestamp,
                    isEmergency = entity.isEmergency,
                    rmsLevel = entity.rmsLevel,
                    localId = entity.id
                )
            }

            val response = apiService.uploadEvents(SoundEventUploadRequest(events = dtos))

            if (response.isSuccessful && response.body()?.success == true) {
                soundEventDao.markAsSynced(unsyncedEvents.map { it.id })
                Pair(response.body()!!.uploadedCount, response.body()!!.failedCount)
            } else {
                Log.e(TAG, "Sync failed")
                Pair(0, unsyncedEvents.size)
            }
        }
    } catch (e: Exception) {
        Log.e(TAG, "Network error during sync: ${e.message}", e)
        Pair(-1, -1)
    }


    /**
     * Delete synced events to free up space
     * @return number of deleted events
     */
    suspend fun deleteSyncedEvents(): Int {
        return soundEventDao.deleteSyncedEvents()
    }

    /**
     * Clear all events (use with caution)
     */
    suspend fun clearAllEvents() {
        soundEventDao.deleteAllEvents()
    }
}