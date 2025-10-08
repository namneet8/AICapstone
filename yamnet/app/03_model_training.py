"""
Step 3: Train Classification Model
Train a neural network on YAMNet embeddings
"""

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class AudioClassifierTrainer:
    def __init__(self, embeddings_path, models_path):
        """
        Initialize trainer
        
        Args:
            embeddings_path: Path to embeddings directory
            models_path: Path to save trained models
        """
        self.embeddings_path = Path(embeddings_path)
        self.models_path = Path(models_path)
        self.finetuned_path = self.models_path / "finetuned"
        self.finetuned_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        with open(self.embeddings_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata["class_names"]
        self.num_classes = len(self.class_names)

    def load_data(self):
        """Load embeddings and labels"""
        print("Loading data...")
        embeddings = np.load(self.embeddings_path / "embeddings.npy")
        labels = np.load(self.embeddings_path / "labels.npy")

        print(f"✓ Loaded {len(embeddings)} samples")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Labels shape: {labels.shape}")
        return embeddings, labels

    def split_data(self, embeddings, labels, test_size=0.15, val_size=0.15):
        """Split data into train, validation, and test sets"""
        print("\nSplitting data...")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            embeddings, labels, test_size=test_size, stratify=labels, random_state=42
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_model(self, input_dim):
        """Build classification model"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, epochs=100, batch_size=32):
        """Train and evaluate the model"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        embeddings, labels = self.load_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(embeddings, labels)

        input_dim = embeddings.shape[1]
        model = self.build_model(input_dim)
        model.summary()

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.finetuned_path / "best_model.h5"),
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

        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ Training complete!")
        model.save(str(self.finetuned_path / "final_model.h5"))
        print(f"✓ Model saved to: {self.finetuned_path / 'final_model.h5'}")

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # Predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, self.class_names)
        self.plot_training_curves(history)
        self.plot_per_class_accuracy(cm, self.class_names)

        # Save metadata
        per_class_acc = (cm.diagonal() / cm.sum(axis=1)).tolist()
        model_metadata = {
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "input_dim": int(input_dim),
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "per_class_accuracy": {
                self.class_names[i]: float(per_class_acc[i])
                for i in range(len(self.class_names))
            }
        }
        with open(self.finetuned_path / "model_metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2)

        history_dict = {
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']]
        }
        with open(self.finetuned_path / "training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Model saved to: {self.finetuned_path}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return model, history

    @staticmethod
    def plot_training_curves(history):
        """Visualize training loss and accuracy"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix - Test Set")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Normalized Confusion Matrix - Test Set")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_per_class_accuracy(cm, class_names):
        """Plot per-class accuracy bar chart"""
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, per_class_acc, color='steelblue', alpha=0.7)
        plt.title("Per-Class Accuracy on Test Set")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    embeddings_path = r"E:/yamnet/data/embeddings"
    models_path = r"E:/yamnet/models"

    trainer = AudioClassifierTrainer(embeddings_path, models_path)
    trainer.train(epochs=100, batch_size=32)


if __name__ == "__main__":
    main()
