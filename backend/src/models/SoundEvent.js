// src/models/SoundEvent.js
import mongoose from 'mongoose';

/**
 * SoundEvent Schema
 * Represents a detected sound event from the Android app
 */
const soundEventSchema = new mongoose.Schema(
  {
    // Sound label (e.g., "Dog bark", "Alarm")
    label: {
      type: String,
      required: [true, 'Label is required'],
      trim: true
    },
    // Detection confidence (0-1)
    confidence: {
      type: Number,
      required: [true, 'Confidence is required'],
      min: [0, 'Confidence must be >= 0'],
      max: [1, 'Confidence must be <= 1']
    },
    // Client-side timestamp (milliseconds since epoch)
    timestamp: {
      type: Number,
      required: [true, 'Timestamp is required']
    },
    // Whether this is an emergency sound
    isEmergency: {
      type: Boolean,
      default: false
    },
    // Root Mean Square audio level
    rmsLevel: {
      type: Number,
      required: [true, 'RMS Level is required'],
      min: 0
    },
    // Local ID from Room database (optional)
    localId: {
      type: Number,
      required: false
    },
    // Server-side timestamp
    serverTimestamp: {
      type: Date,
      default: Date.now
    }
  },
  {
    timestamps: true, // Adds createdAt and updatedAt automatically
    collection: 'sound_events'
  }
);

// Indexes for faster queries
soundEventSchema.index({ timestamp: -1 });
soundEventSchema.index({ isEmergency: 1 });
soundEventSchema.index({ label: 1 });

const SoundEvent = mongoose.model('SoundEvent', soundEventSchema);

export default SoundEvent;