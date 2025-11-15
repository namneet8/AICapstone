// src/controllers/soundEventController.js
import SoundEvent from '../models/SoundEvent.js';

/**
 * Upload batch of sound events
 * POST /api/sound-events/batch
 * 
 * @route   POST /api/sound-events/batch
 * @access  Public
 * @param   {Array} events - Array of sound event objects
 * @returns {Object} Upload result with success status and counts
 */
export const uploadEventsBatch = async (req, res) => {
  try {
    const { events } = req.body;

    // Validate input
    if (!events || !Array.isArray(events) || events.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Events array is required and cannot be empty',
        uploadedCount: 0,
        failedCount: 0
      });
    }

    // Insert multiple documents
    const result = await SoundEvent.insertMany(events, {
      ordered: false // Continue even if some fail
    });

    res.status(201).json({
      success: true,
      message: `${result.length} events saved successfully`,
      uploadedCount: result.length,
      failedCount: events.length - result.length
    });

  } catch (error) {
    console.error('Batch upload error:', error);

    // Handle Mongoose validation errors
    if (error.name === 'ValidationError') {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        error: error.message,
        uploadedCount: 0,
        failedCount: req.body.events?.length || 0
      });
    }

    // Handle other errors
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message,
      uploadedCount: 0,
      failedCount: req.body.events?.length || 0
    });
  }
};

/**
 * Get all events with optional filters
 * GET /api/sound-events
 * 
 * @route   GET /api/sound-events
 * @access  Public
 * @query   {Number} limit - Maximum number of results (default: 100)
 * @query   {Number} skip - Number of results to skip (default: 0)
 * @query   {Boolean} isEmergency - Filter by emergency status
 * @query   {String} label - Filter by sound label
 * @returns {Object} Array of sound events with pagination info
 */
export const getAllEvents = async (req, res) => {
  try {
    const { 
      limit = 100, 
      skip = 0,
      isEmergency,
      label 
    } = req.query;

    // Build filter object
    const filter = {};
    if (isEmergency !== undefined) {
      filter.isEmergency = isEmergency === 'true';
    }
    if (label) {
      filter.label = label;
    }

    // Fetch events with filters
    const events = await SoundEvent.find(filter)
      .sort({ timestamp: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip));

    const total = await SoundEvent.countDocuments(filter);

    res.json({
      success: true,
      data: events,
      pagination: {
        total,
        limit: parseInt(limit),
        skip: parseInt(skip)
      }
    });

  } catch (error) {
    console.error('Error fetching events:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching events',
      error: error.message
    });
  }
};

/**
 * Get event statistics
 * GET /api/sound-events/stats
 * 
 * @route   GET /api/sound-events/stats
 * @access  Public
 * @returns {Object} Statistics including total events, emergencies, and breakdown by label
 */
export const getStats = async (req, res) => {
  try {
    // Run multiple queries in parallel
    const [total, emergencies, byLabel] = await Promise.all([
      SoundEvent.countDocuments(),
      SoundEvent.countDocuments({ isEmergency: true }),
      SoundEvent.aggregate([
        {
          $group: {
            _id: '$label',
            count: { $sum: 1 },
            avgConfidence: { $avg: '$confidence' }
          }
        },
        { $sort: { count: -1 } }
      ])
    ]);

    res.json({
      success: true,
      data: {
        totalEvents: total,
        emergencyEvents: emergencies,
        eventsByLabel: byLabel
      }
    });

  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching statistics',
      error: error.message
    });
  }
};

/**
 * Delete old events
 * DELETE /api/sound-events/cleanup
 * 
 * @route   DELETE /api/sound-events/cleanup
 * @access  Public
 * @query   {Number} daysOld - Delete events older than X days (default: 30)
 * @returns {Object} Number of deleted events
 */
export const cleanupOldEvents = async (req, res) => {
  try {
    const { daysOld = 30 } = req.query;
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - parseInt(daysOld));

    const result = await SoundEvent.deleteMany({
      serverTimestamp: { $lt: cutoffDate }
    });

    res.json({
      success: true,
      message: `${result.deletedCount} events deleted`,
      deletedCount: result.deletedCount
    });

  } catch (error) {
    console.error('Cleanup error:', error);
    res.status(500).json({
      success: false,
      message: 'Error during cleanup',
      error: error.message
    });
  }
};