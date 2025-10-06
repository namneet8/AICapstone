"""
Step 5: Convert Model to TensorFlow Lite (TFLite)
-------------------------------------------------
This script converts a trained Keras model into multiple TFLite formats:
 - Float32 (baseline)
 - Float16 (half-precision)
 - Int8 (quantized with representative dataset)
and evaluates their size, accuracy, and inference speed.

Author: [Your Name]
"""

import tensorflow as tf
import numpy as np
import json
import time
from pathlib import Path
import os
import matplotlib.pyplot as plt


class TFLiteConverter:
    def __init__(self, model_dir, data_dir, output_dir):
        """
        Initialize converter paths.

        Args:
            model_dir: Directory of trained model (.h5 and metadata)
            data_dir: Directory containing embeddings and labels
            output_dir: Directory to save TFLite models
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Conversion
    # -------------------------------------------------------
    def _load_model(self):
        print("\nLoading trained model...")
        model = tf.keras.models.load_model(self.model_dir / "best_model.h5")
        with open(self.model_dir / "model_metadata.json", "r") as f:
            metadata = json.load(f)
        print("✓ Model loaded successfully!")
        model.summary()
        return model, metadata

    def _load_embeddings(self):
        print("\nLoading embeddings for quantization calibration...")
        embeddings = np.load(self.data_dir / "embeddings.npy")
        labels = np.load(self.data_dir / "labels.npy")
        print(f"✓ Loaded {len(embeddings)} embeddings.")
        return embeddings, labels

    def representative_dataset_gen(self, embeddings):
        """Representative dataset for Int8 quantization"""
        num_calibration_samples = min(100, len(embeddings))
        print(f"Using {num_calibration_samples} samples for calibration.")
        for i in range(num_calibration_samples):
            sample = embeddings[i:i+1].astype(np.float32)
            yield [sample]

    def convert_models(self, model, embeddings):
        """Convert to Float32, Float16, and Int8 TFLite models"""
        tflite_models = {}

        # Float32
        print("\nConverting to Float32...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_models["float32"] = converter.convert()
        float32_path = self.output_dir / "audio_classifier_float32.tflite"
        float32_path.write_bytes(tflite_models["float32"])
        print(f"✓ Saved Float32 model: {float32_path}")

        # Float16
        print("\nConverting to Float16...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_models["float16"] = converter.convert()
        float16_path = self.output_dir / "audio_classifier_float16.tflite"
        float16_path.write_bytes(tflite_models["float16"])
        print(f"✓ Saved Float16 model: {float16_path}")

        # Int8
        print("\nConverting to Int8...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self.representative_dataset_gen(embeddings)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_models["int8"] = converter.convert()
        int8_path = self.output_dir / "audio_classifier_int8.tflite"
        int8_path.write_bytes(tflite_models["int8"])
        print(f"✓ Saved Int8 model: {int8_path}")

        return tflite_models, [float32_path, float16_path, int8_path]

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------
    def evaluate_tflite_model(self, tflite_model, X_test, y_test):
        """Compute accuracy for TFLite model"""
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct = 0
        for i in range(len(X_test)):
            input_data = X_test[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])
            if np.argmax(output_data) == y_test[i]:
                correct += 1

        return correct / len(X_test)

    def benchmark_tflite_model(self, tflite_model, X_test, num_trials=100):
        """Benchmark inference latency"""
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        times = []
        for _ in range(num_trials):
            idx = np.random.randint(0, len(X_test))
            input_data = X_test[idx:idx+1].astype(np.float32)
            start = time.time()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])
            times.append((time.time() - start) * 1000)

        return np.mean(times), np.std(times)

    # -------------------------------------------------------
    # Run full pipeline
    # -------------------------------------------------------
    def run(self):
        model, metadata = self._load_model()
        embeddings, labels = self._load_embeddings()

        # Split test set
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            embeddings, labels, test_size=0.15, random_state=42, stratify=labels
        )

        # Convert models
        tflite_models, paths = self.convert_models(model, embeddings)

        # Compare sizes
        keras_size = os.path.getsize(self.model_dir / "best_model.h5") / 1024
        sizes = [len(m) / 1024 for m in tflite_models.values()]
        print(f"\nModel Sizes (KB): Keras={keras_size:.2f}, Float32={sizes[0]:.2f}, Float16={sizes[1]:.2f}, Int8={sizes[2]:.2f}")

        # Evaluate accuracy
        print("\nEvaluating models on test set...")
        acc = {
            "keras": metadata["test_accuracy"],
            "float32": self.evaluate_tflite_model(tflite_models["float32"], X_test, y_test),
            "float16": self.evaluate_tflite_model(tflite_models["float16"], X_test, y_test),
            "int8": self.evaluate_tflite_model(tflite_models["int8"], X_test, y_test)
        }

        # Benchmark inference
        print("\nBenchmarking inference speed...")
        keras_times = []
        for _ in range(100):
            idx = np.random.randint(0, len(X_test))
            sample = X_test[idx:idx+1]
            start = time.time()
            _ = model.predict(sample, verbose=0)
            keras_times.append((time.time() - start) * 1000)
        keras_mean = np.mean(keras_times)

        float32_mean, _ = self.benchmark_tflite_model(tflite_models["float32"], X_test)
        float16_mean, _ = self.benchmark_tflite_model(tflite_models["float16"], X_test)
        int8_mean, _ = self.benchmark_tflite_model(tflite_models["int8"], X_test)

        print(f"\nKeras: {keras_mean:.2f} ms")
        print(f"Float32: {float32_mean:.2f} ms (x{keras_mean/float32_mean:.2f} speedup)")
        print(f"Float16: {float16_mean:.2f} ms (x{keras_mean/float16_mean:.2f} speedup)")
        print(f"Int8: {int8_mean:.2f} ms (x{keras_mean/int8_mean:.2f} speedup)")

        # Save metadata
        metadata_out = {
            "original_model": str(self.model_dir / "best_model.h5"),
            "tflite_models": {
                "float32": {"path": str(paths[0]), "size_kb": sizes[0], "accuracy": acc["float32"], "time_ms": float32_mean},
                "float16": {"path": str(paths[1]), "size_kb": sizes[1], "accuracy": acc["float16"], "time_ms": float16_mean},
                "int8": {"path": str(paths[2]), "size_kb": sizes[2], "accuracy": acc["int8"], "time_ms": int8_mean},
            },
            "class_names": metadata["class_names"]
        }

        with open(self.output_dir / "conversion_metadata.json", "w") as f:
            json.dump(metadata_out, f, indent=2)

        print("\n" + "="*60)
        print("TFLITE CONVERSION COMPLETE!")
        print("="*60)
        print(f"\nResults:\n"
              f"  Keras Acc: {acc['keras']*100:.2f}%\n"
              f"  Float32 Acc: {acc['float32']*100:.2f}%\n"
              f"  Float16 Acc: {acc['float16']*100:.2f}%\n"
              f"  Int8 Acc: {acc['int8']*100:.2f}%\n")

        print("Recommendation:")
        if sizes[2] < 100 and acc["int8"] > 0.9:
            print(" → Use Int8 model for best size/speed tradeoff")
        elif acc["float16"] > acc["int8"] + 0.02:
            print(" → Use Float16 model for better accuracy")
        else:
            print(" → Use Float32 model for maximum accuracy")


def main():
    converter = TFLiteConverter(
        model_dir=r"E:/yamnet/models/finetuned",
        data_dir=r"E:/yamnet/data/embeddings",
        output_dir=r"E:/yamnet/models/tflite"
    )
    converter.run()


if __name__ == "__main__":
    main()
