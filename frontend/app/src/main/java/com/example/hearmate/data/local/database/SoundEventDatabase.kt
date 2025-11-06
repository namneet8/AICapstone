package com.example.hearmate.data.local.database

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.hearmate.data.local.dao.SoundEventDao
import com.example.hearmate.data.local.entity.SoundEventEntity

/**
 * Room Database for HearMate application
 * Stores sound detection events locally
 */
@Database(
    entities = [SoundEventEntity::class],
    version = 1,
    exportSchema = false
)
abstract class SoundEventDatabase : RoomDatabase() {

    /**
     * Provides access to SoundEvent DAO
     */
    abstract fun soundEventDao(): SoundEventDao

    companion object {
        const val DATABASE_NAME = "hearmate_db"
    }
}
