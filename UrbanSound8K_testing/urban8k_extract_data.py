# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 17:34:14 2025

@author: Eric
"""

import pandas as pd
import numpy as np
import librosa
import os
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress librosa warning about file loading
warnings.filterwarnings('ignore', category=UserWarning)

class UrbanSoundFeatureExtractor:
    """
    A comprehensive feature extractor for the UrbanSound8K dataset.
    Supports multiple feature extraction methods and data preprocessing.
    """
    
    def __init__(self, dataset_root_path, sample_rate=22050, n_mfcc=40):
        """
        Initialize the feature extractor.
        
        Args:
            dataset_root_path (str): Root path to UrbanSound8K dataset (should be the parent folder containing both metadata and audio folders)
            sample_rate (int): Target sample rate for audio processing
            n_mfcc (int): Number of MFCC coefficients to extract
        """
        self.dataset_root_path = dataset_root_path
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.metadata_path = os.path.join(dataset_root_path, 'metadata', 'UrbanSound8K.csv')
        self.audio_path = os.path.join(dataset_root_path, 'audio')
        
    def _extract_features_from_file(self, file_path):
        """
        Extract MFCC, Chromagram, and Mel Spectrogram features from an audio file.
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            # Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            # Other features can be added here
            
            # Combine features (e.g., mean and std)
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                [np.mean(zcr)], [np.std(zcr)],
                [np.mean(spectral_centroid)], [np.std(spectral_centroid)]
            ])
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def run_extraction(self, class_filter=None):
        """
        Run the feature extraction process for the specified classes.
        
        Args:
            class_filter (list): List of classes to include. If None, all are included.
        """
        metadata = pd.read_csv(self.metadata_path)
        
        if class_filter:
            metadata = metadata[metadata['class'].isin(class_filter)]
            print(f"Filtering dataset for classes: {class_filter}")
            
        features_list = []
        labels_list = []
        failed_files = 0
        
        # Use tqdm for a progress bar
        print("\nStarting feature extraction...")
        for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
            file_path = os.path.join(self.audio_path, f'fold{row["fold"]}', row['slice_file_name'])
            
            if not os.path.exists(file_path):
                # This is common with incomplete datasets, so just note it
                failed_files += 1
                continue
                
            features = self._extract_features_from_file(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(row['class'])
        
        if not features_list:
            print("No features were extracted. Please check file paths and class filters.")
            return None, None
            
        features_np = np.array(features_list)
        labels_np = np.array(labels_list)

        print("\n=== Feature Extraction Summary ===")
        print(f"Successfully processed: {len(features_list)} files")
        print(f"Failed files: {failed_files}")
        print(f"Feature array shape: {features_np.shape}")
        print(f"Labels array shape: {labels_np.shape}")
        print(f"Unique classes: {np.unique(labels_np)}")
        print("==================================")
        
        return features_np, labels_np

if __name__ == '__main__':
    # Define your dataset root path here
    DATASET_ROOT_PATH = r"E:\UrbanSound8k\UrbanSound8K\UrbanSound8K"
    SAVE_PATH = r'D:\centennial\centennial 2025 fall\comp385\assignment\UrbanSound8K_testing\extracted_features'
    
    # Classes for our specific use case (traffic and emergency sounds)
    TARGET_CLASSES = ['car_horn', 'drilling', 'engine_idling', 'siren', 'street_music', 'jackhammer', 'air_conditioner']

    extractor = UrbanSoundFeatureExtractor(DATASET_ROOT_PATH)
    features, labels = extractor.run_extraction(class_filter=TARGET_CLASSES)
    
    if features is not None and labels is not None:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        np.save(os.path.join(SAVE_PATH, 'urbansound_features.npy'), features)
        np.save(os.path.join(SAVE_PATH, 'urbansound_labels.npy'), labels)
        print(f"Features and labels saved to '{SAVE_PATH}'")

        # Optional: Visualize class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(y=labels, order=pd.Series(labels).value_counts().index)
        plt.title('Distribution of Extracted Classes')
        plt.xlabel('Number of Samples')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.show()