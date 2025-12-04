package com.example.hearmate.core.worker

import android.content.Context
import androidx.work.*
import java.util.concurrent.TimeUnit

/**
 * Initializes and schedules the periodic sync worker.
 * Call this once when the app starts (from Application or MainActivity).
 */
object WorkManagerInitializer {

    private const val TAG = "WorkManagerInit"

    /**
     * Schedules the SyncWorker to run every 1 hour.
     *
     * Configuration:
     * - Runs every 15 minutes
     * - Requires network connectivity
     * - Keeps existing work if already scheduled (no duplicates)
     * - Retries with exponential backoff if sync fails
     *
     * @param context Application context
     */
    fun schedulePeriodicSync(context: Context) {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED) // Only run when network available
            .build()

        val syncRequest = PeriodicWorkRequestBuilder<SyncWorker>(
            repeatInterval = 15, // Every 15 minutes
            repeatIntervalTimeUnit = TimeUnit.MINUTES
        )
            .setConstraints(constraints)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL, // Retry with increasing delays
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .build()

        // Schedule the work, replacing any existing work with the same name
        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            SyncWorker.WORK_NAME,
            ExistingPeriodicWorkPolicy.REPLACE, // Keep existing if already scheduled
            syncRequest
        )

        android.util.Log.d(TAG, "Periodic sync scheduled (every 1 hour)")
    }

    /**
     * Cancels the periodic sync worker.
     * Use this if user wants to disable auto-sync.
     */
    fun cancelPeriodicSync(context: Context) {
        WorkManager.getInstance(context).cancelUniqueWork(SyncWorker.WORK_NAME)
        android.util.Log.d(TAG, "Periodic sync cancelled")
    }
}