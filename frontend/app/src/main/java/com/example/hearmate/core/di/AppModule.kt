package com.example.hearmate.core.di

import android.content.Context
import com.example.hearmate.core.classifier.VineetTrainedModel
import com.example.hearmate.core.interfaces.SoundClassifier
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt Dependency Injection Module for application-wide dependencies.
 *
 * Purpose:
 * --------
 * Provides singleton instances of core app components that are shared
 * across the entire application lifecycle.
 *
 * Currently Provides:
 * - SoundClassifier: The TFLite model for audio classification
 *
 * Note: AudioRecorderManager and SoundEventRepository are @Singleton
 * classes with @Inject constructors, so Hilt provides them automatically.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * Provides the sound classifier implementation.
     *
     * The classifier uses a custom TFLite model (VineetTrainedModel) that
     * detects emergency sounds like sirens, gunshots, glass breaking, etc.
     *
     * @param context Application context (needed to load model from assets)
     * @return SoundClassifier singleton instance
     */
    @Provides
    @Singleton
    fun provideSoundClassifier(
        @ApplicationContext context: Context
    ): SoundClassifier {
        return VineetTrainedModel(context)
    }
}