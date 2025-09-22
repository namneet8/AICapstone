# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:30:55 2025

@author: Eric
"""

import os
import numpy as np
import pandas as pd
import json
import pickle
import librosa
import librosa.feature
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
import wave
import threading
import time
from collections import deque
import argparse

# Configuration
FEATURES_DIR = "E:/GoogleAudioSet/extracted_features"
MODEL_DIR = os.path.join(FEATURES_DIR, "models")

# Audio parameters (same as training)
SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0
SEQUENCE_LENGTH = 32

# Real-time audio parameters
CHUNK = 1024
CHANNELS = 1
RECORD_SECONDS = 2  # Process audio in 2-second chunks
OVERLAP_SECONDS = 0.5  # Overlap between chunks for smoother detection

class AudioPredictor:
    def __init__(self, model_type='CNN_Only', use_tflite=True):
        """
        Initialize the audio predictor
        
        Args:
            model_type: 'CNN_Only' or 'CNN_LSTM'
            use_tflite: Whether to use TensorFlow Lite model for mobile optimization
        """
        self.model_type = model_type
        self.use_tflite = use_tflite
        
        # Load model metadata
        self.load_metadata()
        
        # Load scaler
        self.load_scaler()
        
        # Load model
        self.load_model()
        
        # Audio buffer for real-time processing
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * MAX_DURATION))
        self.is_recording = False
        
    def load_metadata(self):
        """Load model metadata"""
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        self.num_classes = self.metadata['num_classes']
        
        print(f"Loaded model metadata:")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
        
    def load_scaler(self):
        """Load the feature scaler"""
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Loaded feature scaler")
        
    def load_model(self):
        """Load the trained model"""
        if self.use_tflite:
            # Load TensorFlow Lite model
            tflite_path = os.path.join(MODEL_DIR, f'{self.model_type}_quantized.tflite')
            if os.path.exists(tflite_path):
                self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"Loaded TensorFlow Lite model: {tflite_path}")
                print(f"Input shape: {self.input_details[0]['shape']}")
                print(f"Output shape: {self.output_details[0]['shape']}")
            else:
                print(f"TFLite model not found, falling back to full model")
                self.use_tflite = False
        
        if not self.use_tflite:
            # Load full Keras model
            model_path = os.path.join(MODEL_DIR, f'{self.model_type}_full_model.h5')
            self.model = load_model(model_path)
            print(f"Loaded full Keras model: {model_path}")
    
    def extract_features(self, audio_data, sr=SAMPLE_RATE):
        """
        Extract audio features (same as training)
        """
        try:
            # Ensure audio is the right length
            if len(audio_data) < sr * 0.5:  # Less than 0.5 seconds
                return None
                
            # Pad or truncate to MAX_DURATION
            max_len = int(sr * MAX_DURATION)
            if len(audio_data) > max_len:
                audio_data = audio_data[:max_len]
            elif len(audio_data) < max_len:
                audio_data = np.pad(audio_data, (0, max_len - len(audio_data)))
            
            features = []
            
            # 1. MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            features.extend([
                np.mean(spectral_centroid), np.std(spectral_centroid),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            # 3. Mel-frequency features
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=40)
            mel_mean = np.mean(mel_spectrogram, axis=1)
            mel_std = np.std(mel_spectrogram, axis=1)
            features.extend(mel_mean)
            features.extend(mel_std)
            
            # 4. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
            
            # 5. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            features.extend(tonnetz_mean)
            
            # 6. Temporal features
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def preprocess_features(self, features):
        """Preprocess features for prediction"""
        if features is None:
            return None
        
        # Reshape to 2D for scaler
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        if self.model_type == 'CNN_LSTM':
            # Create sequences for CNN+LSTM
            feature_dim = features_scaled.shape[1]
            if feature_dim % SEQUENCE_LENGTH != 0:
                pad_width = SEQUENCE_LENGTH - (feature_dim % SEQUENCE_LENGTH)
                features_scaled = np.pad(features_scaled, ((0, 0), (0, pad_width)), mode='constant')
            
            new_feature_dim = features_scaled.shape[1] // SEQUENCE_LENGTH
            features_scaled = features_scaled.reshape(1, SEQUENCE_LENGTH, new_feature_dim)
        
        return features_scaled
    
    def predict(self, audio_data):
        """
        Predict sound class from audio data
        """
        # Extract features
        features = self.extract_features(audio_data)
        if features is None:
            return None, None
        
        # Preprocess features
        processed_features = self.preprocess_features(features)
        if processed_features is None:
            return None, None
        
        try:
            if self.use_tflite:
                # TensorFlow Lite prediction
                self.interpreter.set_tensor(
                    self.input_details[0]['index'], 
                    processed_features.astype(np.float32)
                )
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            else:
                # Full Keras model prediction
                predictions = self.model.predict(processed_features, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None
    
    def predict_file(self, audio_file_path):
        """
        Predict sound class from an audio file
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
            
            # Predict
            predicted_class, confidence = self.predict(audio_data)
            
            if predicted_class:
                print(f"\nFile: {os.path.basename(audio_file_path)}")
                print(f"Predicted: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                print("-" * 50)
                
                return predicted_class, confidence
            else:
                print(f"Failed to predict for file: {audio_file_path}")
                return None, None
                
        except Exception as e:
            print(f"Error processing file {audio_file_path}: {e}")
            return None, None
    
    def start_real_time_prediction(self, confidence_threshold=0.7):
        """
        Start real-time audio prediction from microphone
        """
        print(f"\nStarting real-time prediction with {self.model_type} model...")
        print(f"Confidence threshold: {confidence_threshold}")
        print("Press Ctrl+C to stop\n")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Open audio stream
            stream = audio.open(
                format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("Listening...")
            self.is_recording = True
            
            # Audio buffer for processing
            buffer_size = int(SAMPLE_RATE * RECORD_SECONDS)
            audio_buffer = np.zeros(buffer_size)
            buffer_pos = 0
            
            while self.is_recording:
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Add to buffer
                chunk_size = len(audio_chunk)
                if buffer_pos + chunk_size <= buffer_size:
                    audio_buffer[buffer_pos:buffer_pos + chunk_size] = audio_chunk
                    buffer_pos += chunk_size
                else:
                    # Buffer is full, process it
                    remaining = buffer_size - buffer_pos
                    audio_buffer[buffer_pos:] = audio_chunk[:remaining]
                    
                    # Predict
                    predicted_class, confidence = self.predict(audio_buffer)
                    
                    if predicted_class and confidence > confidence_threshold:
                        print(f"🚨 DETECTED: {predicted_class} (Confidence: {confidence:.3f})")
                    
                    # Shift buffer for overlap
                    overlap_size = int(SAMPLE_RATE * OVERLAP_SECONDS)
                    audio_buffer[:-overlap_size] = audio_buffer[overlap_size:]
                    audio_buffer[-overlap_size:] = audio_chunk[remaining:remaining + overlap_size]
                    buffer_pos = buffer_size - overlap_size
                    
                    # Add remaining chunk data
                    remaining_chunk = audio_chunk[remaining + overlap_size:]
                    if len(remaining_chunk) > 0:
                        next_size = min(len(remaining_chunk), buffer_size - buffer_pos)
                        audio_buffer[buffer_pos:buffer_pos + next_size] = remaining_chunk[:next_size]
                        buffer_pos += next_size
        
        except KeyboardInterrupt:
            print("\nStopping real-time prediction...")
            self.is_recording = False
        
        finally:
            # Clean up
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
            print("Audio stream closed")
    
    def batch_test_files(self, test_directory, confidence_threshold=0.5):
        """
        Test the model on a batch of audio files
        """
        print(f"\nTesting files in: {test_directory}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("-" * 60)
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        results = []
        
        # Process all audio files in directory
        for root, dirs, files in os.walk(test_directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(root, file)
                    
                    # Get predicted class
                    predicted_class, confidence = self.predict_file(file_path)
                    
                    if predicted_class:
                        results.append({
                            'file': file,
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'high_confidence': confidence >= confidence_threshold
                        })
        
        # Summary statistics
        if results:
            print(f"\nBatch Test Results Summary:")
            print(f"Total files processed: {len(results)}")
            
            high_conf_results = [r for r in results if r['high_confidence']]
            print(f"High confidence predictions: {len(high_conf_results)}")
            
            # Class distribution
            class_counts = {}
            for result in high_conf_results:
                class_name = result['predicted_class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\nHigh confidence detections by class:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
            
            # Average confidence
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"\nAverage confidence: {avg_confidence:.4f}")
        
        return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Audio Sound Classification Predictor')
    parser.add_argument('--model', choices=['CNN_Only', 'CNN_LSTM'], default='CNN_Only',
                       help='Model type to use (default: CNN_Only)')
    parser.add_argument('--tflite', action='store_true', default=True,
                       help='Use TensorFlow Lite model (default: True)')
    parser.add_argument('--mode', choices=['realtime', 'file', 'batch'], default='realtime',
                       help='Prediction mode (default: realtime)')
    parser.add_argument('--file', type=str,
                       help='Audio file path for single file prediction')
    parser.add_argument('--directory', type=str,
                       help='Directory path for batch prediction')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for alerts (default: 0.7)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = AudioPredictor(model_type=args.model, use_tflite=args.tflite)
        print(f"Successfully loaded {args.model} model ({'TFLite' if args.tflite else 'Full Keras'})")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model first by running train_model.py")
        return
    
    # Run prediction based on mode
    if args.mode == 'realtime':
        predictor.start_real_time_prediction(confidence_threshold=args.threshold)
    
    elif args.mode == 'file':
        if not args.file:
            print("Please specify --file for single file prediction")
            return
        
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        predictor.predict_file(args.file)
    
    elif args.mode == 'batch':
        if not args.directory:
            print("Please specify --directory for batch prediction")
            return
        
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}")
            return
        
        predictor.batch_test_files(args.directory, confidence_threshold=args.threshold)

def demo_usage():
    """
    Demonstration of different usage patterns
    """
    print("Audio Classification Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor with CNN_Only model (recommended for mobile)
    predictor = AudioPredictor(model_type='CNN_Only', use_tflite=True)
    
    print("\n1. Single File Prediction Example:")
    # Example: predict a single file
    # predictor.predict_file("path/to/your/audio/file.wav")
    
    print("\n2. Batch Testing Example:")
    # Example: test all files in a directory
    # predictor.batch_test_files("path/to/test/directory")
    
    print("\n3. Real-time Prediction Example:")
    # Example: start real-time prediction
    # predictor.start_real_time_prediction(confidence_threshold=0.7)

def test_specific_file():
    """
    Test a specific audio file - modify this function for quick testing
    """
    # Your specific file path
    test_file = r"E:\GoogleAudioSet\raw_audio\Fire engine, fire truck (siren)\0cOGkwEnSXM_40.wav"
    
    print(f"Testing specific file: {test_file}")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = AudioPredictor(model_type='CNN_Only', use_tflite=True)
        print("Successfully loaded CNN_Only model with TensorFlow Lite")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test the file
    if os.path.exists(test_file):
        predicted_class, confidence = predictor.predict_file(test_file)
        
        if predicted_class:
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"   File: {os.path.basename(test_file)}")
            print(f"   Predicted Class: {predicted_class}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            if confidence > 0.7:
                print(f"   Status: HIGH CONFIDENCE ✅")
            elif confidence > 0.5:
                print(f"   Status: MEDIUM CONFIDENCE ⚠️")
            else:
                print(f"   Status: LOW CONFIDENCE ❌")
        else:
            print("❌ Failed to make prediction")
    else:
        print(f"❌ File not found: {test_file}")

def test_batch_fire_engine_sounds():
    """
    Test all fire engine sounds in your dataset
    """
    fire_engine_dir = r"E:\GoogleAudioSet\raw_audio\Fire engine, fire truck (siren)"
    
    print(f"Testing all fire engine sounds in: {fire_engine_dir}")
    print("=" * 70)
    
    # Initialize predictor
    try:
        predictor = AudioPredictor(model_type='CNN_Only', use_tflite=True)
        print("Successfully loaded CNN_Only model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test all files in the directory
    if os.path.exists(fire_engine_dir):
        results = predictor.batch_test_files(fire_engine_dir, confidence_threshold=0.5)
        
        # Additional analysis for fire engine sounds
        print(f"\n📊 FIRE ENGINE DETECTION ANALYSIS:")
        correct_predictions = [r for r in results if 'fire engine' in r['predicted_class'].lower() or 'siren' in r['predicted_class'].lower()]
        print(f"   Correctly identified as fire engine/siren: {len(correct_predictions)}/{len(results)}")
        
        if results:
            accuracy = len(correct_predictions) / len(results) * 100
            print(f"   Detection accuracy: {accuracy:.1f}%")
    else:
        print(f"❌ Directory not found: {fire_engine_dir}")

if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Quick test options when running without arguments
        print("🔊 Audio Classification Predictor")
        print("=" * 50)
        print("Choose an option:")
        print("1. Test specific file (Fire engine example)")
        print("2. Test all fire engine sounds")
        print("3. See command line usage")
        print()
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            test_specific_file()
        elif choice == '2':
            test_batch_fire_engine_sounds()
        else:
            print("\nCommand line usage examples:")
            print("python predict_sounds.py --mode file --file \"E:\\GoogleAudioSet\\raw_audio\\Fire engine, fire truck (siren)\\0cOGkwEnSXM_40.wav\"")
            print("python predict_sounds.py --mode batch --directory \"E:\\GoogleAudioSet\\raw_audio\\Fire engine, fire truck (siren)\"")
            print("python predict_sounds.py --mode realtime --model CNN_Only --threshold 0.7")
            print("\nFor all options: python predict_sounds.py --help")