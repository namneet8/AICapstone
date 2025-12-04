package com.example.hearmate.core.worker

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.hearmate.data.repository.SoundEventRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

/**
 * Worker that runs periodically (every 1 hour) to sync unsynced events to MongoDB.
 *
 * Features:
 * - Runs in background every hour
 * - Checks network connectivity before syncing
 * - Retries failed syncs automatically
 * - Cleans up synced events to save space
 */
@HiltWorker
class SyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val repository: SoundEventRepository
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        const val WORK_NAME = "sound_event_sync_worker"
        private const val TAG = "SyncWorker"
    }

    /**
     * Main work execution method.
     * Checks network, syncs events, and cleans up synced data.
     *
     * @return Result.success() if sync completed, Result.retry() if network unavailable
     */
    override suspend fun doWork(): Result {
        Log.d(TAG, "Starting periodic sync...")

        // Check if network is available before attempting sync
        if (!isNetworkAvailable()) {
            Log.w(TAG, "No network available, will retry later")
            return Result.retry() // WorkManager will retry automatically
        }

        // Check if there are events to sync
        val unsyncedCount = repository.getUnsyncedCount()
        if (unsyncedCount == 0) {
            Log.d(TAG, "No events to sync")
            return Result.success()
        }

        Log.d(TAG, "Found $unsyncedCount unsynced events, starting sync...")

        // Perform sync to MongoDB
        val (successCount, failureCount) = repository.syncToMongoDB()

        return when {
            // Network error occurred
            failureCount == -1 -> {
                Log.e(TAG, "Network error during sync, will retry")
                Result.retry()
            }

            // All events synced successfully
            successCount > 0 && failureCount == 0 -> {
                Log.d(TAG, "Successfully synced $successCount events")

                // Delete old synced events to free up space (optional)
                val deleted = repository.deleteSyncedEvents()
                Log.d(TAG, "Cleaned up $deleted synced events")

                Result.success()
            }

            // Some events failed to sync
            else -> {
                Log.w(TAG, "Partial sync: $successCount succeeded, $failureCount failed")
                Result.success() // Still mark as success, failed events will retry next time
            }
        }
    }

    /**
     * Checks if the device has an active network connection.
     * Works with WiFi, Cellular, and Ethernet connections.
     *
     * @return true if network is available and connected, false otherwise
     */
    private fun isNetworkAvailable(): Boolean {
        val connectivityManager = applicationContext.getSystemService(
            Context.CONNECTIVITY_SERVICE
        ) as ConnectivityManager

        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false

        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }
}