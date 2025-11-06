// src/routes/soundEvents.js
import express from 'express';
import {
  uploadEventsBatch,
  getAllEvents,
  getStats,
  cleanupOldEvents
} from '../controllers/soundEventController.js';

const router = express.Router();

/**
 * Sound Events Routes
 */

// Upload batch of events
router.post('/batch', uploadEventsBatch);

// Get all events with optional filters
router.get('/', getAllEvents);

// Get event statistics
router.get('/stats', getStats);

// Delete old events
router.delete('/cleanup', cleanupOldEvents);

export default router;