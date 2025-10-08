"""
Master Pipeline Script
Runs complete audio classification pipeline from raw data to deployment
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required libraries are installed"""
    print("Checking requirements...")
    required = {
        'tensorflow': 'tensorflow',
        'tensorflow_hub': 'tensorflow-hub',
        'numpy': 'numpy',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'sklearn': 'scikit-learn',
        'pyaudio': 'pyaudio',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (install with: pip install {package})")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✓ All requirements met")
    return True

def run_preprocessing(raw_path, output_path):
    """Step 1: Preprocess raw audio"""
    print("\n" + "="*70)
    print(" STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    import os
    import shutil
    import librosa
    import soundfile as sf
    import numpy as np
    from pathlib import Path
    
    class AudioSetPreprocessor:
        def __init__(self, raw_audio_path, output_path, target_sr=16000):
            self.raw_audio_path = Path(raw_audio_path)
            self.output_path = Path(output_path)
            self.target_sr = target_sr
            
            self.class_names = [
                "Alarm clock",
                "Explosion",
                "Gunshot, gunfire",
                "Siren",
                "Vehicle horn, car horn, honking"
            ]
            
            self.processed_path = self.output_path / "processed"
            for class_name in self.class_names:
                (self.processed_path / class_name).mkdir(parents=True, exist_ok=True)
        
        def find_audio_files(self):
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(list(self.raw_audio_path.rglob(f'*{ext}')))
            
            return audio_files
        
        def classify_by_filename(self, filename):
            filename_lower = filename.lower()
            
            keywords = {
                "Alarm clock": ["alarm", "clock"],
                "Explosion": ["explosion", "blast", "boom", "explode"],
                "Gunshot, gunfire": ["gun", "shot", "fire", "gunshot", "gunfire"],
                "Siren": ["siren", "ambulance", "police", "emergency"],
                "Vehicle horn, car horn, honking": ["horn", "honk", "car", "vehicle"]
            }
            
            for class_name, keyword_list in keywords.items():
                for keyword in keyword_list:
                    if keyword in filename_lower:
                        return class_name
            
            return None
        
        def process_audio_file(self, file_path, class_name):
            try:
                audio, sr = librosa.load(file_path, sr=None)
                
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                
                output_file = self.processed_path / class_name / file_path.name
                sf.write(output_file, audio, self.target_sr)
                
                return True, len(audio) / self.target_sr
            
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                return False, 0
        
        def run(self):
            print(f"\nSearching for audio files in: {self.raw_audio_path}")
            audio_files = self.find_audio_files()
            print(f"Found {len(audio_files)} audio files")
            
            if len(audio_files) == 0:
                print("\n⚠️  No audio files found!")
                return False
            
            print("\nProcessing files...")
            stats = {class_name: 0 for class_name in self.class_names}
            total_duration = {class_name: 0 for class_name in self.class_names}
            unclassified = 0
            
            for i, file_path in enumerate(audio_files, 1):
                class_name = self.classify_by_filename(file_path.stem)
                
                if class_name:
                    success, duration = self.process_audio_file(file_path, class_name)
                    if success:
                        stats[class_name] += 1
                        total_duration[class_name] += duration
                        if i % 10 == 0:
                            print(f"  Processed {i}/{len(audio_files)} files...")
                else:
                    unclassified += 1
            
            print("\n" + "="*70)
            print("PREPROCESSING SUMMARY")
            print("="*70)
            
            total_processed = sum(stats.values())
            print(f"\nTotal files processed: {total_processed}")
            print(f"Unclassified files: {unclassified}")
            print("\nPer-class statistics:")
            
            for class_name in self.class_names:
                count = stats[class_name]
                duration = total_duration[class_name]
                avg_duration = duration / count if count > 0 else 0
                print(f"  {class_name:30s}: {count:3d} files ({duration:6.1f}s total)")
            
            return total_processed > 0
    
    preprocessor = AudioSetPreprocessor(raw_path, output_path)
    return preprocessor.run()

def run_embedding_extraction(processed_path, output_path):
    """Step 2: Extract YAMNet embeddings"""
    print("\n" + "="*70)
    print(" STEP 2: EMBEDDING EXTRACTION")
    print("="*70)
    
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import librosa
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    class EmbeddingExtractor:
        def __init__(self, processed_path, output_path):
            self.processed_path = Path(processed_path)
            self.output_path = Path(output_path)
            self.embeddings_path = self.output_path / "embeddings"
            self.embeddings_path.mkdir(parents=True, exist_ok=True)
            
            self.class_names = [
                "Alarm clock",
                "Explosion",
                "Gunshot, gunfire",
                "Siren",
                "Vehicle horn, car horn, honking"
            ]
            
            print("Loading YAMNet model...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            print("✓ YAMNet loaded")
            
            self.sample_rate = 16000
        
        def extract_embedding(self, audio_path):
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                min_length = int(self.sample_rate * 0.96)
                if len(audio) < min_length:
                    audio = np.pad(audio, (0, min_length - len(audio)))
                
                audio = audio.astype(np.float32)
                
                _, embeddings, _ = self.yamnet_model(audio)
                avg_embedding = np.mean(embeddings.numpy(), axis=0)
                
                return avg_embedding
            
            except Exception as e:
                return None
        
        def run(self):
            all_embeddings = []
            all_labels = []
            all_filenames = []
            
            for class_idx, class_name in enumerate(self.class_names):
                class_path = self.processed_path / class_name
                
                if not class_path.exists():
                    continue
                
                audio_files = list(class_path.glob("*.wav"))
                print(f"\n{class_name}: {len(audio_files)} files")
                
                if len(audio_files) == 0:
                    continue
                
                for audio_file in tqdm(audio_files, desc=f"  Processing"):
                    embedding = self.extract_embedding(audio_file)
                    
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        all_labels.append(class_idx)
                        all_filenames.append(audio_file.name)
            
            if len(all_embeddings) == 0:
                print("\n⚠️  No embeddings extracted!")
                return False
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            labels_array = np.array(all_labels, dtype=np.int32)
            
            np.save(self.embeddings_path / "embeddings.npy", embeddings_array)
            np.save(self.embeddings_path / "labels.npy", labels_array)
            
            metadata = {
                "class_names": self.class_names,
                "filenames": all_filenames,
                "num_samples": len(all_embeddings),
                "embedding_dim": embeddings_array.shape[1],
                "samples_per_class": {
                    class_name: int(np.sum(labels_array == idx))
                    for idx, class_name in enumerate(self.class_names)
                }
            }
            
            with open(self.embeddings_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("\n" + "="*70)
            print(f"Total embeddings extracted: {len(all_embeddings)}")
            print(f"Embedding dimension: {embeddings_array.shape[1]}")
            
            return True
    
    extractor = EmbeddingExtractor(processed_path, output_path)
    return extractor.run()

def run_training(embeddings_path, models_path):
    """Step 3: Train classification model"""
    print("\n" + "="*70)
    print(" STEP 3: MODEL TRAINING")
    print("="*70)
    
    import tensorflow as tf
    import numpy as np
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    embeddings_path = Path(embeddings_path)
    models_path = Path(models_path)
    finetuned_path = models_path / "finetuned"
    finetuned_path.mkdir(parents=True, exist_ok=True)
    
    with open(embeddings_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    class_names = metadata["class_names"]
    num_classes = len(class_names)
    
    print("\nLoading data...")
    embeddings = np.load(embeddings_path / "embeddings.npy")
    labels = np.load(embeddings_path / "labels.npy")
    print(f"✓ Loaded {len(embeddings)} samples")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/(1-0.2), stratify=y_temp, random_state=42
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Build model
    print("\nBuilding model...")
    input_dim = embeddings.shape[1]
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nAvailable GPUs:", tf.config.list_physical_devices('GPU'))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(finetuned_path / "best_model.h5"),  # ✅ FIXED: Convert Path to str
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    print(f"\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )
    
    model.save(finetuned_path / "final_model.h5")
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test accuracy: {test_accuracy:.4f}")
    
    # Save training history ✅
    with open(finetuned_path / "training_history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    
    # Save metadata
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    per_class_acc = {}
    for idx, class_name in enumerate(class_names):
        mask = y_test == idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
            per_class_acc[class_name] = float(class_acc)
    
    metadata = {
        "class_names": class_names,
        "num_classes": num_classes,
        "input_dim": int(input_dim),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "per_class_accuracy": per_class_acc
    }
    
    with open(finetuned_path / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def run_tflite_conversion(model_path, output_path):
    """Step 4: Convert to TFLite"""
    print("\n" + "="*70)
    print(" STEP 4: TFLITE CONVERSION")
    print("="*70)
    
    import tensorflow as tf
    import json
    from pathlib import Path
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    tflite_path = output_path / "tflite"
    tflite_path.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model...")
    model = tf.keras.models.load_model(model_path / "best_model.h5")
    print("✓ Model loaded")
    
    # Convert to float32
    print("\nConverting to float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    output_file = tflite_path / "audio_classifier_float32.tflite"
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ TFLite model saved: {output_file}")
    print(f"  Size: {len(tflite_model) / 1024:.2f} KB")
    
    # Convert to quantized
    print("\nConverting to quantized TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized = converter.convert()
        
        output_file_q = tflite_path / "audio_classifier_quantized.tflite"
        with open(output_file_q, 'wb') as f:
            f.write(tflite_quantized)
        
        print(f"✓ Quantized model saved: {output_file_q}")
        print(f"  Size: {len(tflite_quantized) / 1024:.2f} KB")
    except Exception as e:
        print(f"⚠️ Quantized conversion failed: {e}")
        output_file_q = None
    
    # Save metadata
    with open(model_path / "model_metadata.json", 'r') as f:
        original_metadata = json.load(f)
    
    conversion_metadata = {
        "class_names": original_metadata["class_names"],
        "num_classes": original_metadata["num_classes"],
        "input_dim": original_metadata["input_dim"],
        "tflite_model": "audio_classifier_float32.tflite",
        "quantized_model": str(output_file_q.name if output_file_q else "conversion_failed"),
        "original_test_accuracy": original_metadata["test_accuracy"]
    }
    
    with open(tflite_path / "conversion_metadata.json", 'w') as f:
        json.dump(conversion_metadata, f, indent=2)
    
    print("✓ Metadata saved")
    return True

def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print(" AUDIO CLASSIFICATION PIPELINE")
    print(" GoogleAudioSet → YAMNet Embeddings → Model Training → TFLite")
    print("="*70)
    
    # Configuration
    RAW_AUDIO_PATH = r"E:/yamnet/data/raw"
    BASE_OUTPUT_PATH = r"E:/yamnet"
    
    print("\nConfiguration:")
    print(f"  Raw audio: {RAW_AUDIO_PATH}")
    print(f"  Output: {BASE_OUTPUT_PATH}")
    
    # Check if paths exist
    if not Path(RAW_AUDIO_PATH).exists():
        print(f"\n✗ Error: Raw audio path does not exist!")
        print(f"  Expected: {RAW_AUDIO_PATH}")
        print("\nPlease update RAW_AUDIO_PATH in the script.")
        return
    
    # Check requirements
    print("\n" + "="*70)
    if not check_requirements():
        return
    
    # User confirmation
    print("\n" + "="*70)
    print("Pipeline will execute the following steps:")
    print("  1. Data Preprocessing (classify and resample audio)")
    print("  2. Embedding Extraction (extract YAMNet features)")
    print("  3. Model Training (train classifier on embeddings)")
    print("  4. TFLite Conversion (convert for deployment)")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return
    
    # Execute pipeline
    try:
        # Step 1: Preprocessing
        data_path = Path(BASE_OUTPUT_PATH) / "data"
        if not run_preprocessing(RAW_AUDIO_PATH, data_path):
            print("\n✗ Preprocessing failed!")
            return
        
        # Step 2: Embedding extraction
        processed_path = data_path / "processed"
        if not run_embedding_extraction(processed_path, data_path):
            print("\n✗ Embedding extraction failed!")
            return
        
        # Step 3: Training
        embeddings_path = data_path / "embeddings"
        models_path = Path(BASE_OUTPUT_PATH) / "models"
        if not run_training(embeddings_path, models_path):
            print("\n✗ Training failed!")
            return
        
        # Step 4: TFLite conversion
        finetuned_path = models_path / "finetuned"
        if not run_tflite_conversion(finetuned_path, models_path):
            print("\n✗ TFLite conversion failed!")
            return
        
        # Success
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE!")
        print("="*70)
        print("\n✓ All steps completed successfully!")
        print("\nGenerated files:")
        print(f"  Processed audio: {data_path / 'processed'}")
        print(f"  Embeddings: {data_path / 'embeddings'}")
        print(f"  Trained models: {models_path / 'finetuned'}")
        print(f"  TFLite models: {models_path / 'tflite'}")
        print("\nYou can now run the real-time classifier!")
        print("  python realtime_classifier.py")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()