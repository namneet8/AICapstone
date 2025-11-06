package com.example.hearmate.core.di

import android.content.Context
import androidx.room.Room
import com.example.hearmate.data.local.database.SoundEventDatabase
import com.example.hearmate.data.local.dao.SoundEventDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Dagger Hilt module for providing database dependencies
 */
@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    /**
     * Provides the Room database instance
     * Singleton ensures only one instance exists
     */
    @Provides
    @Singleton
    fun provideSoundEventDatabase(
        @ApplicationContext context: Context
    ): SoundEventDatabase {
        return Room.databaseBuilder(
            context,
            SoundEventDatabase::class.java,
            SoundEventDatabase.DATABASE_NAME
        )
            .fallbackToDestructiveMigration() // For development - removes this in production
            .build()
    }

    /**
     * Provides the DAO from database
     */
    @Provides
    @Singleton
    fun provideSoundEventDao(database: SoundEventDatabase): SoundEventDao {
        return database.soundEventDao()
    }
}