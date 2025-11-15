import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import soundfile as sf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

# ===============================
# 1. LOAD YAMNET MODEL
# ===============================
print("Loading YAMNet from TF Hub...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
print("YAMNet loaded successfully!")

# ===============================
# 2. AUDIO LOADING FUNCTION
# ===============================
@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert to float32, resample to 16kHz mono."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # Resample if needed
    if sample_rate != 16000:
        # Use basic resampling since we don't have tfio
        current_length = tf.shape(wav)[0]
        target_length = tf.cast(
            tf.cast(current_length, tf.float32) * 16000.0 / tf.cast(sample_rate, tf.float32),
            tf.int32
        )
        wav = tf.image.resize(tf.expand_dims(tf.expand_dims(wav, 0), -1), 
                              [1, target_length], 
                              method='bilinear')
        wav = tf.squeeze(wav)
    
    return wav

# Alternative: Python-based loading (more compatible)
def load_wav_16k_mono_numpy(filepath, target_sr=16000):
    """Load audio file using soundfile, resample to 16kHz mono."""
    audio, sr = sf.read(filepath)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        duration = len(audio) / sr
        target_length = int(duration * target_sr)
        audio = tf.signal.resample(audio, target_length).numpy()
    
    return audio.astype(np.float32)

# ===============================
# 3. PREPARE DATASET
# ===============================
def prepare_dataset(data_dir):
    """
    Prepare dataset by loading audio files and organizing into DataFrame.
    Similar to ESC-50 format in the tutorial.
    """
    data_dir = Path(data_dir)
    
    # Get all class directories
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found classes: {class_names}")
    
    # Create mapping
    map_class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Collect all files
    data_list = []
    for class_name in class_names:
        class_path = data_dir / class_name
        wav_files = list(class_path.glob('*.wav')) + list(class_path.glob('*.wave'))
        
        for wav_file in wav_files:
            data_list.append({
                'filename': str(wav_file),
                'category': class_name,
                'target': map_class_to_id[class_name]
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    print(f"Total audio files: {len(df)}")
    print(f"\nClass distribution:")
    print(df['category'].value_counts())
    
    return df, class_names

# ===============================
# 4. EXTRACT EMBEDDINGS
# ===============================
def extract_embeddings_from_dataset(df):
    """
    Extract YAMNet embeddings for all audio files.
    Returns arrays of embeddings, labels, and file indices.
    """
    all_embeddings = []
    all_labels = []
    all_indices = []
    
    print("\nExtracting embeddings...")
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(df)}...")
        
        try:
            # Load audio
            wav_data = load_wav_16k_mono_numpy(row['filename'])
            
            # Extract embeddings using YAMNet
            scores, embeddings, spectrogram = yamnet_model(wav_data)
            
            # Each audio file produces multiple embeddings (one per frame)
            num_embeddings = embeddings.shape[0]
            
            # Store embeddings with corresponding labels
            all_embeddings.append(embeddings.numpy())
            all_labels.extend([row['target']] * num_embeddings)
            all_indices.extend([idx] * num_embeddings)
            
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.array(all_labels)
    indices_array = np.array(all_indices)
    
    print(f"\nTotal embeddings extracted: {embeddings_array.shape[0]}")
    print(f"Embedding shape: {embeddings_array.shape[1]}")
    
    return embeddings_array, labels_array, indices_array

# ===============================
# 5. CREATE AND TRAIN MODEL
# ===============================
def create_classification_model(num_classes, embedding_size=1024):
    """Create a simple classification model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_size,), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes)
    ], name='hearmate_classifier')
    
    return model

def train_model(embeddings, labels, indices, num_classes, epochs=20, batch_size=32):
    """Train the classification model on embeddings."""
    
    # Split by file index to avoid data leakage
    # Files from the same audio shouldn't be in both train and val
    unique_indices = np.unique(indices)
    train_indices, val_indices = train_test_split(
        unique_indices, test_size=0.2, random_state=42
    )
    
    # Create train/val masks
    train_mask = np.isin(indices, train_indices)
    val_mask = np.isin(indices, val_indices)
    
    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create and compile model
    model = create_classification_model(num_classes)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

# ===============================
# 6. CREATE END-TO-END MODEL
# ===============================
class ReduceMeanLayer(tf.keras.layers.Layer):
    """Custom layer to apply reduce_mean with a fixed name."""
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

def create_serving_model(trained_model, class_names):
    """
    Create end-to-end model that takes raw audio as input.
    This combines YAMNet with the trained classifier.
    """
    # Input: raw audio waveform
    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    
    # YAMNet embedding extraction (frozen)
    embedding_extraction_layer = hub.KerasLayer(
        yamnet_model_handle,
        trainable=False,
        name='yamnet'
    )
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    
    # Our trained classifier
    serving_outputs = trained_model(embeddings_output)
    
    # Average predictions across all frames
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    
    # Create model
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    
    return serving_model

# ===============================
# 7. MAIN TRAINING PIPELINE
# ===============================
def main():
    # Configuration
    DATA_DIR = 'raw/'
    SAVE_DIR = './hearmate_model'
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # 1. Prepare dataset
    print("="*50)
    print("STEP 1: Preparing Dataset")
    print("="*50)
    df, class_names = prepare_dataset(DATA_DIR)
    
    # 2. Extract embeddings
    print("\n" + "="*50)
    print("STEP 2: Extracting YAMNet Embeddings")
    print("="*50)
    embeddings, labels, indices = extract_embeddings_from_dataset(df)
    
    # 3. Train classifier
    print("\n" + "="*50)
    print("STEP 3: Training Classifier")
    print("="*50)
    trained_model, history = train_model(
        embeddings, labels, indices, 
        num_classes=len(class_names),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # 4. Create serving model
    print("\n" + "="*50)
    print("STEP 4: Creating End-to-End Model")
    print("="*50)
    serving_model = create_serving_model(trained_model, class_names)
    
    # 5. Save models
    print("\n" + "="*50)
    print("STEP 5: Saving Models")
    print("="*50)
    
    # Save serving model (for inference)
    os.makedirs(SAVE_DIR, exist_ok=True)
    serving_model.save(f'{SAVE_DIR}/serving_model', include_optimizer=False)
    print(f"Serving model saved to: {SAVE_DIR}/serving_model")
    
    # Save class names
    with open(f'{SAVE_DIR}/class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Class names saved to: {SAVE_DIR}/class_names.txt")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/training_history.png')
    print(f"Training history plot saved to: {SAVE_DIR}/training_history.png")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return serving_model, class_names, history

# ===============================
# 8. TEST FUNCTION
# ===============================
def test_model(model_path, audio_file, class_names_file):
    """Test the saved model on a single audio file."""
    # Load model
    model = tf.saved_model.load(model_path)
    
    # Load class names
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Load audio
    audio = load_wav_16k_mono_numpy(audio_file)
    
    # Run inference
    predictions = model(audio)
    predicted_class_idx = tf.math.argmax(predictions).numpy()
    predicted_class = class_names[predicted_class_idx]
    
    # Get probabilities
    probabilities = tf.nn.softmax(predictions).numpy()
    
    print(f"\nPrediction for: {audio_file}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {probabilities[predicted_class_idx]:.4f}")
    print("\nAll probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob:.4f}")
    
    return predicted_class, probabilities

if __name__ == "__main__":
    # Train the model
    serving_model, class_names, history = main()
    
    # Optional: Test on a sample file
    # Uncomment and modify the path to test
    # test_model(
    #     './hearmate_model/serving_model',
    #     '/path/to/test/audio.wav',
    #     './hearmate_model/class_names.txt'
    # )