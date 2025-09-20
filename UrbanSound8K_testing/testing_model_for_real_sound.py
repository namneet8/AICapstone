# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 19:11:38 2025

@author: Eric
"""

import joblib
import librosa
import numpy as np

# Load the trained model, scaler, and label encoder
# Make sure the paths to your saved models are correct
try:
    model = joblib.load('./saved_models/best_model.joblib')
    scaler = joblib.load('./saved_models/scaler.joblib')
    label_encoder = joblib.load('./saved_models/label_encoder.joblib')
    print("All necessary model components loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not find a model file. Please ensure the path is correct.")
    print(f"Details: {e}")
    exit()

def preprocess_audio(audio_path, sample_rate=22050):
    """
    Loads an audio file and extracts the same features used during model training.
    
    The number of MFCCs is set to 16 to match the feature count
    that the saved StandardScaler was trained on.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Replicate the feature extraction logic from your original code
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_features = np.mean(librosa.power_to_db(mel_spec), axis=1)

    # Corrected MFCC count to 16 to match the scaler's expected features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
    mfcc_features = np.mean(mfcc, axis=1)

    spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    combined_features = np.concatenate([
        mel_features, mfcc_features,
        [spectral_centroids, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
    ])

    # The model expects a 2D array, so reshape the single feature vector
    return combined_features.reshape(1, -1)


# Path to the new audio file.
# The `r` prefix creates a "raw string", which prevents Python from
# interpreting backslashes as special characters.
new_audio_path = r'E:\UrbanSound8k\UrbanSound8K\UrbanSound8K\audio\fold2\123688-8-0-17.wav'

# Preprocess the new audio file
features_vector = preprocess_audio(new_audio_path)

# Check if feature extraction was successful
if features_vector is not None:
    # Scale the new features using the saved scaler
    features_scaled = scaler.transform(features_vector)

    # Make the prediction
    predicted_label_id = model.predict(features_scaled)[0]

    # Decode the prediction back to a class name
    predicted_class = label_encoder.inverse_transform([predicted_label_id])[0]

    print(f"The detected sound is: {predicted_class}")

