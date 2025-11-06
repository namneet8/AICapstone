package com.example.hearmate.data.local.dao

import androidx.room.*
import com.example.hearmate.data.local.entity.SoundEventEntity
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for sound events
 * Provides database operations for SoundEventEntity
 */
@Dao
interface SoundEventDao {

    /**
     * Insert a new sound event
     * @return the ID of the inserted event
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEvent(event: SoundEventEntity): Long

    /**
     * Insert multiple events
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEvents(events: List<SoundEventEntity>)

    /**
     * Get all events ordered by timestamp (most recent first)
     */
    @Query("SELECT * FROM sound_events ORDER BY timestamp DESC")
    fun getAllEvents(): Flow<List<SoundEventEntity>>

    /**
     * Get all unsynced events (to be uploaded to MongoDB)
     */
    @Query("SELECT * FROM sound_events WHERE isSynced = 0 ORDER BY timestamp ASC")
    suspend fun getUnsyncedEvents(): List<SoundEventEntity>

    /**
     * Get only emergency events
     */
    @Query("SELECT * FROM sound_events WHERE isEmergency = 1 ORDER BY timestamp DESC")
    fun getEmergencyEvents(): Flow<List<SoundEventEntity>>

    /**
     * Get events within a time range
     */
    @Query("SELECT * FROM sound_events WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getEventsByTimeRange(startTime: Long, endTime: Long): List<SoundEventEntity>

    /**
     * Mark events as synced
     */
    @Query("UPDATE sound_events SET isSynced = 1 WHERE id IN (:eventIds)")
    suspend fun markAsSynced(eventIds: List<Long>)

    /**
     * Delete synced events (to free up space)
     */
    @Query("DELETE FROM sound_events WHERE isSynced = 1")
    suspend fun deleteSyncedEvents(): Int

    /**
     * Delete all events
     */
    @Query("DELETE FROM sound_events")
    suspend fun deleteAllEvents()

    /**
     * Get total count of events
     */
    @Query("SELECT COUNT(*) FROM sound_events")
    suspend fun getEventCount(): Int
}