"""
Real-time Audio Classification Application
Classifies live audio with visual feedback
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pyaudio
import json
import time
from collections import deque
import threading
import tkinter as tk
from tkinter import ttk
import librosa

class RealtimeAudioClassifier:
    def __init__(self, tflite_model_path='C:/Users/arsht/Downloads/audio_classifier/models/tflite/audio_classifier_float32.tflite',
                 confidence_threshold=0.6):
        """
        Initialize real-time audio classifier
        
        Args:
            tflite_model_path: Path to TFLite model
            confidence_threshold: Minimum confidence to display prediction
        """
        print("="*60)
        print("INITIALIZING REAL-TIME AUDIO CLASSIFIER")
        print("="*60)
        
        # Load TFLite model
        print("\n1. Loading TFLite model...")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("   ✓ TFLite model loaded")
        
        # Load YAMNet for embeddings
        print("\n2. Loading YAMNet...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("   ✓ YAMNet loaded")
        
        # Load metadata
        print("\n3. Loading model metadata...")
        with open('C:/Users/arsht/Downloads/audio_classifier/models/tflite/conversion_metadata.json', 'r') as f:
            metadata = json.load(f)
        self.class_names = metadata['class_names']
        print(f"   ✓ Classes: {self.class_names}")
        
        # Audio parameters (YAMNet requirements)
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.confidence_threshold = confidence_threshold
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size)
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=3)
        
        # Status flags
        self.is_running = False
        self.last_detection = None
        self.last_detection_time = None
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE!")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk duration: {self.chunk_duration}s")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Number of classes: {len(self.class_names)}")
    
    def extract_embedding(self, audio_data):
        """Extract YAMNet embedding from audio"""
        # Ensure audio is float32 and normalized
        audio_data = audio_data.astype(np.float32)
        
        # Pad if too short
        if len(audio_data) < self.sample_rate * 0.96:  # YAMNet minimum
            audio_data = np.pad(audio_data, (0, int(self.sample_rate * 0.96) - len(audio_data)))
        
        # Extract embeddings
        _, embeddings, _ = self.yamnet_model(audio_data)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings.numpy(), axis=0)
        return avg_embedding.reshape(1, -1).astype(np.float32)
    
    def predict(self, embedding):
        """Run prediction on embedding"""
        self.interpreter.set_tensor(self.input_details[0]['index'], embedding)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]
    
    def smooth_predictions(self, probabilities):
        """Smooth predictions using history"""
        self.prediction_history.append(probabilities)
        
        if len(self.prediction_history) < 2:
            return probabilities
        
        # Average recent predictions
        smoothed = np.mean(self.prediction_history, axis=0)
        return smoothed
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio buffer and make predictions"""
        while self.is_running:
            if len(self.audio_buffer) >= self.chunk_size:
                # Get audio chunk
                audio_chunk = np.array(list(self.audio_buffer))
                
                try:
                    # Extract embedding
                    embedding = self.extract_embedding(audio_chunk)
                    
                    # Predict
                    probabilities = self.predict(embedding)
                    
                    # Smooth predictions
                    smoothed_probs = self.smooth_predictions(probabilities)
                    
                    # Get top prediction
                    predicted_class = np.argmax(smoothed_probs)
                    confidence = smoothed_probs[predicted_class]
                    
                    # Check if above threshold
                    if confidence > self.confidence_threshold:
                        class_name = self.class_names[predicted_class]
                        
                        # Check if it's a new detection (or enough time has passed)
                        current_time = time.time()
                        if (self.last_detection != class_name or 
                            self.last_detection_time is None or 
                            current_time - self.last_detection_time > 2.0):
                            
                            self.last_detection = class_name
                            self.last_detection_time = current_time
                            
                            # Update GUI
                            self.update_detection(class_name, confidence, smoothed_probs)
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
            else:
                time.sleep(0.05)
    
    def update_detection(self, class_name, confidence, probabilities):
        """Update GUI with detection"""
        if hasattr(self, 'gui'):
            self.gui.update_detection(class_name, confidence, probabilities)
    
    def start(self):
        """Start audio stream and processing"""
        print("\nStarting audio capture...")
        
        self.is_running = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        print("✓ Audio stream started")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("✓ Processing thread started")
    
    def stop(self):
        """Stop audio stream and processing"""
        print("\nStopping classifier...")
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        
        print("✓ Classifier stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.audio.terminate()


class AudioClassifierGUI:
    def __init__(self, classifier):
        """Initialize GUI"""
        self.classifier = classifier
        self.classifier.gui = self
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Real-time Audio Classifier")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎵 Real-time Audio Classifier",
            font=("Arial", 20, "bold"),
            pady=20
        )
        title_label.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="Listening...",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#666'
        )
        self.status_label.pack()
        
        # Detection frame
        self.detection_frame = tk.Frame(self.root, bg='white', padx=20, pady=20)
        self.detection_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.detection_label = tk.Label(
            self.detection_frame,
            text="No detection yet",
            font=("Arial", 24, "bold"),
            bg='white',
            fg='#999'
        )
        self.detection_label.pack(pady=20)
        
        self.confidence_label = tk.Label(
            self.detection_frame,
            text="",
            font=("Arial", 16),
            bg='white',
            fg='#666'
        )
        self.confidence_label.pack()
        
        # Probability bars frame
        self.probs_frame = tk.Frame(self.root, padx=20, pady=10)
        self.probs_frame.pack(fill='x', padx=20)
        
        self.prob_labels = {}
        self.prob_bars = {}
        
        for class_name in classifier.class_names:
            frame = tk.Frame(self.probs_frame)
            frame.pack(fill='x', pady=5)
            
            label = tk.Label(frame, text=class_name, width=15, anchor='w')
            label.pack(side='left')
            self.prob_labels[class_name] = label
            
            canvas = tk.Canvas(frame, height=20, bg='#e0e0e0')
            canvas.pack(side='left', fill='x', expand=True)
            self.prob_bars[class_name] = canvas
        
        # Control buttons
        button_frame = tk.Frame(self.root, pady=10)
        button_frame.pack()
        
        self.stop_button = tk.Button(
            button_frame,
            text="Stop",
            command=self.stop_classifier,
            font=("Arial", 12),
            padx=20,
            pady=10,
            bg='#ff6b6b',
            fg='white',
            cursor='hand2'
        )
        self.stop_button.pack()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_detection(self, class_name, confidence, probabilities):
        """Update GUI with new detection"""
        # Update detection label
        self.detection_label.config(
            text=f"🔊 {class_name.upper()}",
            fg='#2ecc71' if confidence > 0.8 else '#f39c12'
        )
        
        self.confidence_label.config(
            text=f"Confidence: {confidence*100:.1f}%"
        )
        
        # Flash effect
        self.detection_frame.config(bg='#e8f8f5')
        self.root.after(500, lambda: self.detection_frame.config(bg='white'))
        
        # Update probability bars
        for idx, class_name in enumerate(self.classifier.class_names):
            prob = probabilities[idx]
            canvas = self.prob_bars[class_name]
            
            # Clear canvas
            canvas.delete('all')
            
            # Draw bar
            width = canvas.winfo_width()
            bar_width = int(width * prob)
            
            color = '#2ecc71' if prob > 0.6 else '#3498db' if prob > 0.3 else '#95a5a6'
            canvas.create_rectangle(0, 0, bar_width, 20, fill=color, outline='')
            canvas.create_text(5, 10, text=f"{prob*100:.1f}%", anchor='w', fill='white' if prob > 0.2 else 'black')
    
    def stop_classifier(self):
        """Stop the classifier"""
        self.classifier.stop()
        self.status_label.config(text="Stopped", fg='red')
        self.stop_button.config(state='disabled')
    
    def on_closing(self):
        """Handle window close"""
        self.classifier.cleanup()
        self.root.destroy()
    
    def run(self):
        """Start GUI main loop"""
        self.root.mainloop()


def main():
    """Main function"""
    print("\n" + "="*60)
    print("REAL-TIME AUDIO CLASSIFICATION")
    print("="*60)
    
    # Create classifier
    classifier = RealtimeAudioClassifier(
        tflite_model_path='C:/Users/arsht/Downloads/audio_classifier/models/tflite/audio_classifier_float32.tflite',
        confidence_threshold=0.6
    )
    
    # Start classifier
    classifier.start()
    
    # Create and run GUI
    gui = AudioClassifierGUI(classifier)
    gui.run()


if __name__ == "__main__":
    main()