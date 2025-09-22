# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:25:48 2025

@author: Eric
"""

import os
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Reshape, TimeDistributed, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
FEATURES_DIR = "E:/GoogleAudioSet/extracted_features"
MODEL_DIR = os.path.join(FEATURES_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model parameters for mobile optimization
SEQUENCE_LENGTH = 32  # Reduced for mobile efficiency
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

def load_data():
    """Load the extracted features and labels"""
    print("Loading extracted features...")
    
    # Load training data
    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'))
    y_train = np.load(os.path.join(FEATURES_DIR, 'train_labels.npy'))
    
    # Load test data
    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'))
    y_test = np.load(os.path.join(FEATURES_DIR, 'test_labels.npy'))
    
    # Load class names
    class_names = np.load(os.path.join(FEATURES_DIR, 'class_names.npy'))
    
    # Load feature info
    with open(os.path.join(FEATURES_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    return X_train, X_test, y_train, y_test, class_names, feature_info

def preprocess_data(X_train, X_test, y_train, y_test, num_classes):
    """Preprocess data for training"""
    print("Preprocessing data...")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # For CNN + LSTM, we need to reshape data to create sequences
    # Since we have static features, we'll create artificial sequences
    def create_sequences(X, seq_len):
        """Create overlapping sequences from feature vectors"""
        if len(X.shape) == 2:
            # Pad or truncate features to make them divisible by seq_len
            feature_dim = X.shape[1]
            if feature_dim % seq_len != 0:
                pad_width = seq_len - (feature_dim % seq_len)
                X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
            
            # Reshape to sequences
            new_feature_dim = X.shape[1] // seq_len
            X_reshaped = X.reshape(X.shape[0], seq_len, new_feature_dim)
            return X_reshaped
        return X
    
    # Create sequences for CNN+LSTM
    X_train_seq = create_sequences(X_train_scaled, SEQUENCE_LENGTH)
    X_test_seq = create_sequences(X_test_scaled, SEQUENCE_LENGTH)
    
    print(f"Reshaped training data: {X_train_seq.shape}")
    print(f"Reshaped test data: {X_test_seq.shape}")
    
    return (X_train_scaled, X_test_scaled, X_train_seq, X_test_seq, 
            y_train_cat, y_test_cat, scaler)

def create_cnn_lstm_model(input_shape, num_classes):
    """
    Create a mobile-optimized CNN + LSTM hybrid model
    Designed to be lightweight for mobile deployment
    """
    print(f"Creating CNN + LSTM model with input shape: {input_shape}")
    
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layers for temporal modeling
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN-only model (more mobile-friendly alternative)
    """
    print(f"Creating CNN-only model with input shape: {input_shape}")
    
    # Reshape input for CNN (add sequence dimension)
    input_layer = Input(shape=(input_shape,))
    reshaped = Reshape((input_shape, 1))(input_layer)
    
    # CNN layers
    x = Conv1D(filters=32, kernel_size=5, activation='relu')(reshaped)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def train_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train the model with callbacks"""
    print(f"\nTraining {model_name} model...")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name} model...")
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_confusion_matrix.png'))
    plt.show()
    
    return accuracy

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_training_history.png'))
    plt.show()

def convert_to_tflite(model, model_name):
    """Convert model to TensorFlow Lite for mobile deployment"""
    print(f"\nConverting {model_name} to TensorFlow Lite...")
    
    try:
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization for mobile
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantization for smaller model size
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = os.path.join(MODEL_DIR, f'{model_name}_quantized.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved: {tflite_path}")
        
        # Get model size
        model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        print(f"Model size: {model_size:.2f} MB")
        
        return tflite_path
        
    except Exception as e:
        print(f"Error converting {model_name} to TFLite: {e}")
        return None

def main():
    """Main training function"""
    print("Audio Classification Model Training")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test, class_names, feature_info = load_data()
    num_classes = len(class_names)
    
    # Preprocess data
    (X_train_scaled, X_test_scaled, X_train_seq, X_test_seq, 
     y_train_cat, y_test_cat, scaler) = preprocess_data(
        X_train, X_test, y_train, y_test, num_classes
    )
    
    # Save scaler for inference
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train both models for comparison
    models_to_train = [
        {
            'name': 'CNN_LSTM',
            'model_func': create_cnn_lstm_model,
            'input_shape': X_train_seq.shape[1:],
            'X_train': X_train_seq,
            'X_test': X_test_seq
        },
        {
            'name': 'CNN_Only',
            'model_func': create_cnn_model,
            'input_shape': X_train_scaled.shape[1],
            'X_train': X_train_scaled,
            'X_test': X_test_scaled
        }
    ]
    
    results = {}
    
    for model_config in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_config['name']} Model")
        print(f"{'='*60}")
        
        # Create model
        model = model_config['model_func'](
            model_config['input_shape'], num_classes
        )
        
        # Train model
        history = train_model(
            model,
            model_config['X_train'],
            model_config['X_test'],
            y_train_cat,
            y_test_cat,
            model_config['name']
        )
        
        # Evaluate model
        accuracy = evaluate_model(
            model, 
            model_config['X_test'], 
            y_test_cat, 
            class_names, 
            model_config['name']
        )
        
        # Plot training history
        plot_training_history(history, model_config['name'])
        
        # Convert to TensorFlow Lite for mobile deployment
        tflite_path = convert_to_tflite(model, model_config['name'])
        
        # Save full model
        model.save(os.path.join(MODEL_DIR, f'{model_config["name"]}_full_model.h5'))
        
        results[model_config['name']] = {
            'accuracy': accuracy,
            'model': model,
            'tflite_path': tflite_path
        }
    
    # Compare results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.4f} accuracy")
    
    # Recommend best model for mobile deployment
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nRecommended model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Save model metadata
    model_metadata = {
        'class_names': class_names.tolist(),
        'num_classes': num_classes,
        'feature_dim': X_train_scaled.shape[1],
        'sequence_length': SEQUENCE_LENGTH,
        'input_shape_cnn_lstm': list(X_train_seq.shape[1:]),
        'input_shape_cnn': X_train_scaled.shape[1],
        'results': {name: {'accuracy': float(res['accuracy'])} 
                   for name, res in results.items()}
    }
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\nAll models and metadata saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()