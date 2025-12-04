package com.example.hearmate.di

import android.content.Context
import android.os.Build
import androidx.annotation.RequiresApi
import com.example.hearmate.core.service.ServiceManager
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt Dependency Injection Module for Service-related dependencies.
 *
 * Purpose:
 * --------
 * Provides the ServiceManager singleton that manages the ListeningService lifecycle.
 * Separated from AppModule for better organization of service-layer dependencies.
 *
 * Why Singleton?
 * - ServiceManager holds the service binding state
 * - Multiple instances would cause binding conflicts
 * - ViewModel and Activity need to share the same instance
 */
@Module
@InstallIn(SingletonComponent::class)
object ServiceModule {

    /**
     * Provides the ServiceManager singleton.
     *
     * ServiceManager acts as a bridge between UI layer (ViewModel) and
     * the ListeningService, handling:
     * - Service start/stop
     * - Binding/unbinding
     * - State synchronization
     *
     * @param context Application context for service operations
     * @return ServiceManager singleton instance
     */
    @RequiresApi(Build.VERSION_CODES.O)
    @Provides
    @Singleton
    fun provideServiceManager(
        @ApplicationContext context: Context
    ): ServiceManager {
        return ServiceManager(context)
    }
}