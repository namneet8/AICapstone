package com.example.hearmate.core.di

import EricTrainedModel
import NameetTrainedModel
import android.content.Context
import com.example.hearmate.core.audio.MockSoundClassifier
import com.example.hearmate.core.audio.SoundClassifier
import com.example.hearmate.core.audio.YamnetSoundClassifier
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    // Model selection: MOCK, YAMNET, or TRAINED
    private enum class ModelType {
        MOCK,
        YAMNET,
        NAMEET_TRAINED,
        ERIC_TRAINED
    }

    private val SELECTED_MODEL = ModelType.NAMEET_TRAINED  // model selected

    @Provides
    @Singleton
    fun provideSoundClassifier(
        @ApplicationContext context: Context
    ): SoundClassifier {
        return when (SELECTED_MODEL) {
            ModelType.ERIC_TRAINED -> EricTrainedModel(context)
            ModelType.YAMNET -> YamnetSoundClassifier(context)
            ModelType.NAMEET_TRAINED -> NameetTrainedModel(context)
            ModelType.MOCK -> MockSoundClassifier()
        }
    }
}