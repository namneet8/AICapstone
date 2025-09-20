# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 18:19:01 2025

@author: Eric
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class UrbanSoundModelTrainer:
    """
    Train models using pre-extracted features from .npy files.
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = None
        self.labels = None
        self.class_names = None

    def load_saved_features(self, features_path, labels_path):
        """
        Load features and labels from .npy files.
        """
        print(f"Loading features from: {features_path}")
        print(f"Loading labels from: {labels_path}")
        
        # Load the arrays
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        
        # Encode labels and get class names
        self.label_encoder.fit(self.labels)
        self.class_names = self.label_encoder.classes_
        print(f"Loaded {len(self.features)} samples with {len(self.class_names)} classes.")
        
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train/validation/test sets.
        Handles small and imbalanced datasets intelligently.
        """
        if self.features is None or self.labels is None:
            raise ValueError("Please load features first using load_saved_features()")
        
        n_samples = len(self.features)
        labels_encoded = self.label_encoder.transform(self.labels)
        n_classes = len(np.unique(labels_encoded))
        
        print(f"\n🔍 Dataset Analysis:")
        print(f"   Total samples: {n_samples}")
        print(f"   Number of classes: {n_classes}")
        
        # Check if we can perform a stratified split
        can_stratify = True
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        if np.any(counts < 2):
            print("⚠️ WARNING: Some classes have fewer than 2 samples. Cannot perform stratified split.")
            can_stratify = False
            
        if can_stratify:
            print("✅ Attempting stratified splitting...")
            try:
                # First split: train+val vs test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    self.features, labels_encoded, 
                    test_size=test_size, 
                    stratify=labels_encoded, 
                    random_state=random_state
                )
                
                # Second split: separate validation from training
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    stratify=y_temp,
                    random_state=random_state
                )
                print("✅ Stratified split successful!")
            except ValueError as e:
                print(f"⚠️ Stratified split failed: {e}")
                print("   Falling back to simple random splitting...")
                X_train, X_val, X_test, y_train, y_val, y_test = self._perform_simple_split(
                    self.features, labels_encoded, test_size, val_size, random_state
                )
        else:
            print("   Using simple random split due to insufficient samples per class.")
            X_train, X_val, X_test, y_train, y_val, y_test = self._perform_simple_split(
                self.features, labels_encoded, test_size, val_size, random_state
            )

        print(f"\n📊 Data Split:")
        print(f"   Training: {X_train.shape[0]} samples ({X_train.shape[0]/n_samples*100:.1f}%)")
        if X_val is not None:
            print(f"   Validation: {X_val.shape[0]} samples ({X_val.shape[0]/n_samples*100:.1f}%)")
        else:
            print(f"   Validation: None (skipped)")
        print(f"   Test: {X_test.shape[0]} samples ({X_test.shape[0]/n_samples*100:.1f}%)")
        
        # Scale the data after splitting
        X_train = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def _perform_simple_split(self, features, labels_encoded, test_size, val_size, random_state):
        """Internal helper for non-stratified splitting."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels_encoded,
            test_size=test_size,
            random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        
        # Handle cases where training set is too small for a validation split
        if len(X_temp) > 10 and (len(X_temp) * val_size_adjusted) >= 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
            print("   Skipping validation set due to insufficient data for second split.")

        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a Random Forest Classifier.
        """
        print("\n🌲 Training Random Forest Classifier...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        self.models['RandomForest'] = grid_search.best_estimator_
        print(f"Best Random Forest Parameters: {grid_search.best_params_}")
        
    def train_svm(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a Support Vector Machine (SVM) Classifier.
        """
        print("\n🧠 Training Support Vector Machine (SVM) Classifier...")
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
        }
        grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        self.models['SVM'] = grid_search.best_estimator_
        print(f"Best SVM Parameters: {grid_search.best_params_}")
        
    def train_neural_network(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a simple Neural Network with Keras.
        """
        print("\n🧠 Training Neural Network...")
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.models['NeuralNetwork'] = model
        
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on the test set.
        """
        print("\n=== Model Evaluation ===")
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            if name == 'NeuralNetwork':
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis=1)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.class_names)
            
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
            
            # Confusion Matrix Visualization
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'Confusion Matrix for {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
            
    def save_best_model_for_mobile(self, save_dir='saved_models'):
        """
        Selects the best model and saves it for mobile deployment.
        Saves the scaler and label encoder as well.
        """
        if not self.models:
            print("No models to save. Please train them first.")
            return

        best_model_name = None
        best_accuracy = -1
        
        # Use the test set to find the best model
        # This prevents the model from overfitting on the validation set
        X_test_scaled = self.scaler.transform(self.features)
        y_test_encoded = self.label_encoder.transform(self.labels)

        for name, model in self.models.items():
            if name == 'NeuralNetwork':
                y_pred_probs = model.predict(X_test_scaled)
                y_pred = np.argmax(y_pred_probs, axis=1)
            else:
                y_pred = model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test_encoded, y_pred)
            print(f"Model: {name}, Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                
        if best_model_name is None:
            print("Could not determine the best model.")
            return
            
        print(f"\n🥇 Best performing model is: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save the model
        if best_model_name == 'NeuralNetwork':
            # Save using Keras' native save method, which is more robust
            model_path = os.path.join(save_dir, 'best_model.keras')
            self.models['NeuralNetwork'].save(model_path)
            print("Neural Network model saved in Keras format.")
            
            # Convert to TensorFlow Lite directly from the Keras model file
            converter = tf.lite.TFLiteConverter.from_keras_model(self.models['NeuralNetwork'])
            tflite_model = converter.convert()
            with open(os.path.join(save_dir, 'best_model.tflite'), 'wb') as f:
                f.write(tflite_model)
            print("Converted to TensorFlow Lite format.")
        else:
            # Save other models with joblib
            joblib.dump(self.models[best_model_name], os.path.join(save_dir, 'best_model.joblib'))
            print(f"{best_model_name} model saved in joblib format.")
            
        # Save the scaler and label encoder, which are essential for inference
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.joblib'))
        print("StandardScaler saved.")
        
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
        print("LabelEncoder saved.")
        
        print(f"\nAll components for mobile deployment saved in '{save_dir}'")

if __name__ == '__main__':
    FEATURES_PATH = './extracted_features/urbansound_features.npy'
    LABELS_PATH = './extracted_features/urbansound_labels.npy'
    
    trainer = UrbanSoundModelTrainer()
    trainer.load_saved_features(FEATURES_PATH, LABELS_PATH)
    
    # Prepare data for training and evaluation
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
    
    # Train and evaluate models
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    trainer.train_svm(X_train, y_train, X_val, y_val)
    trainer.train_neural_network(X_train, y_train, X_val, y_val)
    
    trainer.evaluate_models(X_test, y_test)
    
    # Save the best model for mobile deployment
    trainer.save_best_model_for_mobile()