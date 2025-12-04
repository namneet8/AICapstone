package com.example.hearmate

import android.app.Application
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import androidx.work.WorkManager
import com.example.hearmate.core.worker.WorkManagerInitializer
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

/**
 * Application class for HearMate.
 *
 * Responsibilities:
 * -----------------
 * 1. Initialize Hilt dependency injection
 * 2. Configure WorkManager with Hilt support
 * 3. Schedule periodic background sync task
 *
 * Background Sync:
 * ----------------
 * WorkManager runs a periodic task (every 1 hour) to sync local sound
 * event data to the cloud (MongoDB). This ensures data backup even
 * when the app isn't actively being used.
 *
 * Why @HiltAndroidApp?
 * - Triggers Hilt's code generation
 * - Must be on the Application class
 * - Enables @AndroidEntryPoint on Activities/Services
 */
@HiltAndroidApp
class HearMateApplication : Application() {

    /**
     * Hilt-aware WorkerFactory.
     * Allows WorkManager workers to use @Inject dependencies.
     */
    @Inject
    lateinit var workerFactory: HiltWorkerFactory

    override fun onCreate() {
        super.onCreate()

        // Initialize WorkManager with Hilt support
        // This allows SyncWorker to inject SoundEventRepository
        WorkManager.initialize(
            this,
            Configuration.Builder()
                .setWorkerFactory(workerFactory)
                .build()
        )

        // Schedule periodic sync (runs every 1 hour)
        WorkManagerInitializer.schedulePeriodicSync(this)
    }
}