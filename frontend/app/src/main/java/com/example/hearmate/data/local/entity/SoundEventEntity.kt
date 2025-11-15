package com.example.hearmate.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room Entity representing a detected sound event
 * Stores all classification results locally before syncing to MongoDB
 */
@Entity(tableName = "sound_events")
data class SoundEventEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    val label: String,              // Detected sound label (e.g., "Siren", "Alarm")
    val confidence: Float,          // Classification confidence (0.0 - 1.0)
    val timestamp: Long,            // Detection timestamp in milliseconds
    val isEmergency: Boolean,       // Whether this is an emergency sound
    val rmsLevel: Double,           // Audio RMS level at detection
    val isSynced: Boolean = false   // Whether synced to MongoDB
)