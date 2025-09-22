# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 23:57:11 2025

@author: Eric
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.feature
from tqdm import tqdm
import glob
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
AUDIOSOURCE_ROOT = "E:/GoogleAudioSet"
RAW_AUDIO_DIR = os.path.join(AUDIOSOURCE_ROOT, "raw_audio")
FEATURES_DIR = os.path.join(AUDIOSOURCE_ROOT, "extracted_features")
os.makedirs(FEATURES_DIR, exist_ok=True)

LABELS_TO_EXTRACT = [
    'Alarm clock', 'Siren', 'Fire engine, fire truck (siren)',
    'Car passing by', 'Vehicle horn, car horn, honking',
    'Motor vehicle (road)'
]

# Audio processing parameters
SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0  # seconds

def extract_audio_features(file_path, sr=SAMPLE_RATE):
    """
    Extract comprehensive audio features from a WAV file
    Returns a feature vector combining multiple audio characteristics
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr, duration=MAX_DURATION)
        
        # Ensure minimum length
        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return None
            
        # Extract various features
        features = []
        
        # 1. MFCC features (most important for audio classification)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])
        
        # 3. Mel-frequency features
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_mean = np.mean(mel_spectrogram, axis=1)
        mel_std = np.std(mel_spectrogram, axis=1)
        features.extend(mel_mean)
        features.extend(mel_std)
        
        # 4. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)
        
        # 5. Tonnetz features (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        features.extend(tonnetz_mean)
        
        # 6. Temporal features
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_audio_files():
    """
    Process all WAV files in the raw_audio directory and extract features
    """
    print("Starting feature extraction from WAV files...")
    
    all_features = []
    all_labels = []
    
    # Process each class directory
    for class_name in LABELS_TO_EXTRACT:
        class_dir = os.path.join(RAW_AUDIO_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found for class '{class_name}': {class_dir}")
            continue
            
        # Get all WAV files in this class directory
        wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
        
        if not wav_files:
            print(f"Warning: No WAV files found in {class_dir}")
            continue
            
        print(f"\nProcessing {len(wav_files)} files for class: {class_name}")
        
        # Process each WAV file
        for wav_file in tqdm(wav_files, desc=f"Extracting features for {class_name}"):
            features = extract_audio_features(wav_file)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(class_name)
            else:
                print(f"Failed to extract features from: {wav_file}")
    
    if not all_features:
        print("No features extracted! Please check your audio files.")
        return None, None
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"\nFeature extraction completed!")
    print(f"Total samples: {len(features_array)}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Classes found: {np.unique(labels_array)}")
    
    # Print class distribution
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\nClass distribution:")
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} samples")
    
    return features_array, labels_array

def create_train_test_split(features, labels, test_ratio=0.2):
    """
    Create train/test split while maintaining class distribution
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_ratio, 
        random_state=42, 
        stratify=labels
    )
    
    return X_train, X_test, y_train, y_test

def save_features_and_labels(features, labels):
    """
    Save extracted features and labels to files
    """
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        features, labels_encoded
    )
    
    # Save training data
    np.save(os.path.join(FEATURES_DIR, 'train_features.npy'), X_train)
    np.save(os.path.join(FEATURES_DIR, 'train_labels.npy'), y_train)
    
    # Save test data
    np.save(os.path.join(FEATURES_DIR, 'test_features.npy'), X_test)
    np.save(os.path.join(FEATURES_DIR, 'test_labels.npy'), y_test)
    
    # Save label encoder for future use
    import pickle
    with open(os.path.join(FEATURES_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save class names mapping
    class_names = label_encoder.classes_
    np.save(os.path.join(FEATURES_DIR, 'class_names.npy'), class_names)
    
    print(f"\nSaved features to {FEATURES_DIR}:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Number of classes: {len(class_names)}")
    
    # Save feature info
    feature_info = {
        'feature_dim': X_train.shape[1],
        'num_classes': len(class_names),
        'class_names': class_names.tolist(),
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'sample_rate': SAMPLE_RATE,
        'max_duration': MAX_DURATION
    }
    
    import json
    with open(os.path.join(FEATURES_DIR, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)

if __name__ == "__main__":
    print("Audio Feature Extraction for Machine Learning")
    print("=" * 50)
    
    # Check if librosa is installed
    try:
        import librosa
    except ImportError:
        print("Error: librosa is not installed. Please install it with:")
        print("pip install librosa")
        exit(1)
    
    # Extract features from all WAV files
    features, labels = process_audio_files()
    
    if features is not None:
        # Save the extracted features
        save_features_and_labels(features, labels)
        print(f"\nFeature extraction completed successfully!")
        print(f"Files saved to: {FEATURES_DIR}")
    else:
        print("Feature extraction failed!")