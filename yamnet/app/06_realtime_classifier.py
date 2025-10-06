"""
Step 6: Real-time Audio Classification Application
Enhanced version with auto model selection, smarter buffering, and adaptive confidence threshold.
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
from pathlib import Path


class RealtimeAudioClassifier:
    def __init__(self, tflite_model_path=None, confidence_threshold=None):
        """
        Initialize real-time audio classifier.
        Automatically selects the best available TFLite model.

        Args:
            tflite_model_path: Optional manual model path
            confidence_threshold: Optional manual confidence threshold
        """
        print("=" * 60)
        print("INITIALIZING REAL-TIME AUDIO CLASSIFIER")
        print("=" * 60)

        # Auto-detect model
        model_dir = Path(r"E:/yamnet/models/tflite")
        if tflite_model_path is None:
            # Load metadata to choose the best model
            meta_path = model_dir / "conversion_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                # Prefer int8 > float16 > float32 if available
                for key in ["int8", "float16", "float32"]:
                    if Path(meta["tflite_models"][key]["path"]).exists():
                        tflite_model_path = meta["tflite_models"][key]["path"]
                        print(f"→ Auto-selected best model: {key.upper()} ({tflite_model_path})")
                        break
            else:
                tflite_model_path = model_dir / "audio_classifier_float32.tflite"
                print(f"⚠ No metadata found — using default: {tflite_model_path}")

        # Load TFLite model
        print("\n1. Loading TFLite model...")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✓ Model loaded: {Path(tflite_model_path).name}")
        except Exception as e:
            raise RuntimeError(f"✗ Failed to load TFLite model: {e}")

        # Load YAMNet embeddings model
        print("\n2. Loading YAMNet (TF Hub)...")
        try:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("✓ YAMNet loaded")
        except Exception as e:
            raise RuntimeError(f"✗ Error loading YAMNet: {e}")

        # Load metadata and class names
        print("\n3. Loading metadata...")
        metadata_path = Path(tflite_model_path).parent / "conversion_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            self.class_names = meta["class_names"]
            print(f"✓ Classes loaded: {self.class_names}")
        else:
            print("⚠ No metadata found — using fallback classes.")
            self.class_names = [
                "Alarm clock", "Explosion", "Gunshot, gunfire", "Siren", "Vehicle horn, car horn, honking"
            ]

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.audio_buffer = deque(maxlen=self.chunk_size)
        self.prediction_history = deque(maxlen=4)

        # Confidence threshold adjustment
        if confidence_threshold is None:
            if "int8" in str(tflite_model_path):
                self.confidence_threshold = 0.5
            elif "float16" in str(tflite_model_path):
                self.confidence_threshold = 0.55
            else:
                self.confidence_threshold = 0.6
        else:
            self.confidence_threshold = confidence_threshold

        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.last_detection = None
        self.last_detection_time = None

        print("\nConfiguration:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_duration:.2f}s)")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Classes: {len(self.class_names)}")
        print("=" * 60)

    # ------------------------ Audio Handling ------------------------

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if status:
            print(f"Audio warning: {status}")
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)

    def extract_embedding(self, audio_data):
        """Extract YAMNet embedding"""
        try:
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data /= np.max(np.abs(audio_data))

            # Pad to 0.96s minimum for YAMNet
            min_len = int(self.sample_rate * 0.96)
            if len(audio_data) < min_len:
                audio_data = np.pad(audio_data, (0, min_len - len(audio_data)))

            _, embeddings, _ = self.yamnet_model(audio_data)
            return np.mean(embeddings.numpy(), axis=0).reshape(1, -1).astype(np.float32)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def predict(self, embedding):
        """Run TFLite prediction"""
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], embedding)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def smooth_predictions(self, probabilities):
        """Temporal smoothing"""
        self.prediction_history.append(probabilities)
        return np.mean(self.prediction_history, axis=0)

    def process_audio(self):
        """Main audio processing loop"""
        while self.is_running:
            if len(self.audio_buffer) >= self.chunk_size:
                audio_chunk = np.array(list(self.audio_buffer))
                embedding = self.extract_embedding(audio_chunk)
                if embedding is None:
                    time.sleep(0.05)
                    continue

                probs = self.predict(embedding)
                if probs is None:
                    continue

                probs = self.smooth_predictions(probs)
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]

                if confidence > self.confidence_threshold:
                    class_name = self.class_names[pred_class]
                    current_time = time.time()

                    # Prevent spam
                    if (self.last_detection != class_name or
                        self.last_detection_time is None or
                        current_time - self.last_detection_time > 2.0):
                        self.last_detection = class_name
                        self.last_detection_time = current_time
                        self.update_detection(class_name, confidence, probs)
            time.sleep(0.05)

    def start(self):
        """Start microphone stream"""
        print("🎙 Starting microphone capture...")
        self.is_running = True
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        threading.Thread(target=self.process_audio, daemon=True).start()
        print("✓ Audio stream active")

    def stop(self):
        """Stop audio processing"""
        print("\nStopping audio classifier...")
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("✓ Stopped")

    def cleanup(self):
        """Release resources"""
        self.stop()
        self.audio.terminate()

    def update_detection(self, class_name, confidence, probabilities):
        """Update GUI (called from audio thread)"""
        if hasattr(self, "gui"):
            self.gui.update_detection(class_name, confidence, probabilities)


# ------------------------ GUI Section ------------------------

class AudioClassifierGUI:
    def __init__(self, classifier):
        """Tkinter GUI setup"""
        self.classifier = classifier
        self.classifier.gui = self
        self.root = tk.Tk()
        self.root.title("🔊 Real-time Audio Classifier")
        self.root.geometry("720x600")
        self.root.configure(bg="#f9f9f9")

        # Title
        tk.Label(
            self.root, text="🎵 Real-time Audio Classifier", font=("Arial", 20, "bold"), bg="#f9f9f9"
        ).pack(pady=20)

        # Status
        self.status_label = tk.Label(self.root, text="Listening...", font=("Arial", 14), fg="#27ae60", bg="#f9f9f9")
        self.status_label.pack()

        # Detection
        self.detection_label = tk.Label(self.root, text="Waiting for detection...", font=("Arial", 24, "bold"), bg="white", fg="#95a5a6", relief="solid", bd=1, padx=10, pady=10)
        self.detection_label.pack(pady=20, ipadx=10, ipady=10, fill="x", padx=30)

        self.conf_label = tk.Label(self.root, text="", font=("Arial", 16), bg="#f9f9f9", fg="#7f8c8d")
        self.conf_label.pack()

        # Probability bars
        self.bar_frame = tk.Frame(self.root, bg="#f9f9f9")
        self.bar_frame.pack(fill="x", padx=30, pady=10)

        self.bars = {}
        for cls in classifier.class_names:
            frame = tk.Frame(self.bar_frame, bg="#f9f9f9")
            frame.pack(fill="x", pady=4)
            tk.Label(frame, text=cls, width=25, anchor="w", bg="#f9f9f9").pack(side="left")
            canvas = tk.Canvas(frame, height=25, bg="#ecf0f1", highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True)
            self.bars[cls] = canvas

        # Stop button
        tk.Button(self.root, text="⏹ Stop", command=self.stop, bg="#e74c3c", fg="white", font=("Arial", 12, "bold")).pack(pady=15)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_detection(self, class_name, confidence, probabilities):
        """Update GUI display"""
        self.detection_label.config(text=f"Detected: {class_name.upper()}", fg="#27ae60" if confidence > 0.8 else "#f39c12")
        self.conf_label.config(text=f"Confidence: {confidence * 100:.1f}%")

        for cls, prob in zip(self.classifier.class_names, probabilities):
            canvas = self.bars[cls]
            canvas.delete("all")
            width = canvas.winfo_width() or 200
            bar_width = int(width * prob)
            color = "#27ae60" if prob > 0.6 else "#3498db" if prob > 0.3 else "#95a5a6"
            canvas.create_rectangle(0, 0, bar_width, 25, fill=color)
            canvas.create_text(10, 12, text=f"{prob * 100:.1f}%", anchor="w", fill="white" if prob > 0.15 else "#2c3e50")

    def stop(self):
        self.classifier.stop()
        self.status_label.config(text="Stopped", fg="#e74c3c")

    def on_close(self):
        self.classifier.cleanup()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ------------------------ Main ------------------------

def main():
    print("=" * 60)
    print("REAL-TIME AUDIO CLASSIFICATION APP")
    print("=" * 60)

    try:
        classifier = RealtimeAudioClassifier()
        classifier.start()
        gui = AudioClassifierGUI(classifier)
        gui.run()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
