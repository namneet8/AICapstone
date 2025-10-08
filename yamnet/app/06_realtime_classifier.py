"""
Step 6: Real-time Audio Classification
--------------------------------------
Classifies audio in real-time using microphone input or audio files
Provides both CLI and API interfaces for easy integration
"""

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import json
import queue
import threading
import time
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional, Callable


class AudioClassifierAPI:
    """
    User-friendly API for audio classification
    Can be used in other projects by importing this class
    """
    
    def __init__(self, model_path: str, metadata_path: str, use_tflite: bool = True):
        """
        Initialize the audio classifier
        
        Args:
            model_path: Path to model (.tflite or .h5)
            metadata_path: Path to metadata JSON containing class names
            use_tflite: Whether to use TFLite (True) or Keras (False) model
            
        Example:
            classifier = AudioClassifierAPI(
                model_path="models/tflite/audio_classifier_int8.tflite",
                metadata_path="models/finetuned/model_metadata.json"
            )
        """
        self.use_tflite = use_tflite
        self.sample_rate = 16000
        self.chunk_duration = 5.0
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.class_names = metadata["class_names"]
        self.num_classes = len(self.class_names)
        
        # Load YAMNet
        print("Loading YAMNet model...")
        self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load classifier
        if use_tflite:
            print(f"Loading TFLite model from {model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            print(f"Loading Keras model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.interpreter = None
        
        print("✓ Audio Classifier initialized!")
        print(f"  Classes: {', '.join(self.class_names)}")
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract YAMNet embedding from audio waveform
        
        Args:
            audio: Audio waveform (numpy array, mono, any sample rate)
            
        Returns:
            1024-dimensional embedding vector
        """
        audio = audio.astype(np.float32)
        _, embeddings, _ = self.yamnet(audio)
        return np.mean(embeddings.numpy(), axis=0)
    
    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """
        Classify an audio embedding
        
        Args:
            embedding: 1024-dimensional YAMNet embedding
            
        Returns:
            Class probabilities (numpy array of shape [num_classes])
        """
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], embedding)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            predictions = self.model.predict(embedding, verbose=0)[0]
        
        return predictions
    
    def classify_audio(self, audio: np.ndarray, sample_rate: int = None) -> Dict:
        """
        Classify audio waveform (end-to-end)
        
        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Sample rate of audio (if None, assumes 16kHz)
            
        Returns:
            Dictionary with:
                - predicted_class: Most likely class name
                - confidence: Confidence score (0-1)
                - probabilities: All class probabilities
                - class_scores: List of (class_name, probability) tuples sorted by score
        
        Example:
            result = classifier.classify_audio(audio_data, sample_rate=16000)
            print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
        """
        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Extract embedding and predict
        embedding = self.extract_embedding(audio)
        probabilities = self.predict(embedding)
        
        # Get results
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create sorted class scores
        class_scores = [(self.class_names[i], float(probabilities[i])) 
                       for i in range(self.num_classes)]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "class_scores": class_scores
        }
    
    def classify_file(self, audio_path: str) -> Dict:
        """
        Classify an audio file
        
        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)
            
        Returns:
            Same as classify_audio()
            
        Example:
            result = classifier.classify_file("audio/dog_bark.wav")
            print(f"Top 3 predictions:")
            for class_name, prob in result['class_scores'][:3]:
                print(f"  {class_name}: {prob:.2%}")
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.classify_audio(audio, sr)
    
    def classify_batch(self, audio_files: List[str], 
                      progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Classify multiple audio files
        
        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of classification results
            
        Example:
            files = ["audio1.wav", "audio2.wav", "audio3.wav"]
            results = classifier.classify_batch(files, 
                                               progress_callback=lambda i,n: print(f"{i}/{n}"))
        """
        results = []
        for i, path in enumerate(audio_files):
            result = self.classify_file(path)
            result["file"] = path
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(audio_files))
        
        return results


class RealtimeAudioClassifier:
    """Real-time audio classification from microphone"""
    
    def __init__(self, api: AudioClassifierAPI, smoothing: int = 5):
        """
        Args:
            api: AudioClassifierAPI instance
            smoothing: Number of predictions to average for smoothing
        """
        self.api = api
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.prediction_history = deque(maxlen=smoothing)
        self.callbacks = []
        
    def add_callback(self, callback: Callable[[Dict], None]):
        """
        Add a callback function to receive predictions
        
        Args:
            callback: Function that takes a prediction dict
            
        Example:
            def on_prediction(result):
                print(f"Detected: {result['predicted_class']}")
            
            realtime.add_callback(on_prediction)
        """
        self.callbacks.append(callback)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Internal callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Process audio chunks from queue"""
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.is_running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk.flatten()])
                
                # Process when we have enough samples
                if len(audio_buffer) >= self.api.chunk_samples:
                    audio_segment = audio_buffer[:self.api.chunk_samples]
                    audio_buffer = audio_buffer[self.api.chunk_samples // 2:]  # 50% overlap
                    
                    # Classify
                    result = self.api.classify_audio(audio_segment)
                    
                    # Smooth predictions
                    self.prediction_history.append(result['probabilities'])
                    smoothed_probs = np.mean(self.prediction_history, axis=0)
                    
                    # Update result with smoothed values
                    result['probabilities'] = smoothed_probs.tolist()
                    predicted_idx = np.argmax(smoothed_probs)
                    result['predicted_class'] = self.api.class_names[predicted_idx]
                    result['confidence'] = float(smoothed_probs[predicted_idx])
                    
                    # Update class scores
                    result['class_scores'] = [
                        (self.api.class_names[i], float(smoothed_probs[i])) 
                        for i in range(self.api.num_classes)
                    ]
                    result['class_scores'].sort(key=lambda x: x[1], reverse=True)
                    
                    # Display
                    print(f"\rPredicted: {result['predicted_class']:15s} | "
                          f"Confidence: {result['confidence']:.2%} | "
                          f"Top-3: {', '.join([f'{c}({p:.0%})' for c,p in result['class_scores'][:3]])}",
                          end="", flush=True)
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        callback(result)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError processing audio: {e}")
    
    def start(self):
        """Start real-time classification"""
        print("\n" + "="*70)
        print("REAL-TIME AUDIO CLASSIFICATION")
        print("="*70)
        print(f"Sample Rate: {self.api.sample_rate} Hz")
        print(f"Chunk Duration: {self.api.chunk_duration}s")
        print(f"Classes: {', '.join(self.api.class_names)}")
        print("\nPress Ctrl+C to stop...")
        print("="*70 + "\n")
        
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.api.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.api.sample_rate * 0.5)  # 0.5s blocks
            ):
                while self.is_running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.is_running = False
            process_thread.join(timeout=2)
            print("✓ Classification stopped")
    
    def stop(self):
        """Stop real-time classification"""
        self.is_running = False


# ============================================================================
# CLI Interface and Examples
# ============================================================================

def example_classify_file(api: AudioClassifierAPI):
    """Example: Classify a single audio file"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Classify Single File")
    print("="*70)
    
    audio_file = input("Enter path to audio file: ").strip()
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return
    
    print(f"\nClassifying {audio_file}...")
    result = api.classify_file(audio_file)
    
    print(f"\n✓ Results:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\n  Top 5 Predictions:")
    for i, (class_name, prob) in enumerate(result['class_scores'][:5], 1):
        print(f"    {i}. {class_name:20s} {prob:.2%}")


def example_classify_batch(api: AudioClassifierAPI):
    """Example: Classify multiple files"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Classification")
    print("="*70)
    
    folder = input("Enter folder containing audio files: ").strip()
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
        audio_files.extend(folder_path.glob(ext))
    
    if not audio_files:
        print(f"❌ No audio files found in {folder}")
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    print("Classifying...")
    
    results = api.classify_batch(
        [str(f) for f in audio_files],
        progress_callback=lambda i, n: print(f"\rProgress: {i}/{n}", end="", flush=True)
    )
    
    print("\n\n✓ Batch classification complete!")
    print("\nSummary:")
    for result in results:
        print(f"  {Path(result['file']).name:30s} → {result['predicted_class']:15s} ({result['confidence']:.2%})")


def example_realtime(api: AudioClassifierAPI):
    """Example: Real-time classification"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Real-time Classification")
    print("="*70)
    
    # Optional: Add custom callback
    def log_high_confidence(result):
        if result['confidence'] > 0.8:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] High confidence detection: {result['predicted_class']}")
    
    realtime = RealtimeAudioClassifier(api, smoothing=5)
    realtime.add_callback(log_high_confidence)
    realtime.start()


def main():
    """Main CLI interface"""
    print("\n" + "="*70)
    print("AUDIO CLASSIFIER - USER API")
    print("="*70)
    
    # Initialize API
    model_type = input("\nSelect model type (float32/float16/int8) [int8]: ").strip() or "int8"
    
    tflite_path = Path(f"E:/yamnet/models/tflite/audio_classifier_{model_type}.tflite")
    metadata_path = Path("E:/yamnet/models/finetuned/model_metadata.json")
    
    if not tflite_path.exists():
        print(f"❌ Model not found: {tflite_path}")
        print("Please run 05_tflite_conversion.py first")
        return
    
    # Create API instance
    api = AudioClassifierAPI(
        model_path=str(tflite_path),
        metadata_path=str(metadata_path),
        use_tflite=True
    )
    
    # Menu
    while True:
        print("\n" + "="*70)
        print("Select an option:")
        print("  1. Classify single audio file")
        print("  2. Batch classify folder")
        print("  3. Real-time classification (microphone)")
        print("  4. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            example_classify_file(api)
        elif choice == "2":
            example_classify_batch(api)
        elif choice == "3":
            example_realtime(api)
        elif choice == "4":
            print("\n✓ Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    # You'll need: pip install sounddevice librosa tensorflow tensorflow-hub
    main()