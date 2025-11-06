package com.example.hearmate.core.di

import com.example.hearmate.data.local.dao.SoundEventDao
import com.example.hearmate.data.remote.ApiService
import com.example.hearmate.data.repository.SoundEventRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Dagger Hilt module for providing repository dependencies
 */
@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {

    /**
     * Provides SoundEventRepository
     * Injects both local DAO and remote API service
     */
    @Provides
    @Singleton
    fun provideSoundEventRepository(
        dao: SoundEventDao,
        apiService: ApiService
    ): SoundEventRepository {
        return SoundEventRepository(dao, apiService)
    }
}