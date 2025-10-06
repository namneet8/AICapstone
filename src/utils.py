"""
Utility functions for audio classification project
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
import os

def load_audio(file_path, sr=16000, duration=None):
    """
    Load audio file
    
    Args:
        file_path: Path to audio file
        sr: Sample rate (default 16000 for YAMNet)
        duration: Maximum duration in seconds
        
    Returns:
        audio_data: Audio as numpy array
        sample_rate: Sample rate
    """
    audio, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    return audio, sr

def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio

def plot_waveform(audio, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title="Spectrogram"):
    """Plot audio spectrogram"""
    plt.figure(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def create_directory_structure():
    """Create project directory structure"""
    directories = [
        '../data/raw',
        '../data/processed',
        '../data/embeddings',
        '../models/yamnet',
        '../models/finetuned',
        '../models/tflite',
        '../notebooks',
        '../src',
        '../app'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_audio_duration(file_path):
    """Get audio file duration in seconds"""
    audio, sr = librosa.load(file_path, sr=None)
    return len(audio) / sr

def split_audio_into_chunks(audio, sr, chunk_duration=5.0, overlap=0.5):
    """
    Split audio into overlapping chunks
    
    Args:
        audio: Audio data
        sr: Sample rate
        chunk_duration: Chunk duration in seconds
        overlap: Overlap ratio (0-1)
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
    
    # Handle remainder
    if len(audio) > chunk_samples and len(audio) % hop_samples != 0:
        last_chunk = audio[-chunk_samples:]
        chunks.append(last_chunk)
    elif len(audio) < chunk_samples:
        # Pad if too short
        padded = np.pad(audio, (0, chunk_samples - len(audio)), mode='constant')
        chunks.append(padded)
    
    return chunks

class AudioAugmentor:
    """Audio augmentation utilities"""
    
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    @staticmethod
    def time_shift(audio, sr, shift_max=0.2):
        """Shift audio in time"""
        shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
        return np.roll(audio, shift)
    
    @staticmethod
    def pitch_shift(audio, sr, n_steps=2):
        """Shift pitch"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(audio, rate=1.1):
        """Stretch/compress time"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    @staticmethod
    def change_volume(audio, factor=1.2):
        """Change volume"""
        return audio * factor

def compute_audio_features(audio, sr):
    """
    Compute audio features for analysis
    
    Returns:
        Dictionary of features
    """
    features = {
        'duration': len(audio) / sr,
        'rms_energy': float(np.mean(librosa.feature.rms(y=audio))),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))),
        'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    }
    return features

def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (estimated): {total_params * 4 / 1024:.2f} KB")

def visualize_class_distribution(labels, class_names, title="Class Distribution"):
    """Visualize class distribution"""
    from collections import Counter
    
    class_counts = Counter(labels)
    counts = [class_counts[i] for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color='steelblue', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    
    return class_weights