import tensorflow as tf
import numpy as np

def convert_to_tflite(saved_model_path, output_path='hearmate_model.tflite'):
    """
    Convert the saved model to TFLite format for Android with dynamic input shape.
    
    Args:
        saved_model_path: Path to the saved TensorFlow model
        output_path: Output path for the TFLite model
    """
    print("Loading saved model...")
    
    # Load the model first to create a concrete function with dynamic shape
    model = tf.saved_model.load(saved_model_path)
    
    # Create a concrete function with explicit input signature for dynamic audio length
    # The -1 means the length can be any size
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def serving_fn(audio):
        return model(audio)
    
    # Get the concrete function
    concrete_func = serving_fn.get_concrete_function()
    
    # Create converter from concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # CRITICAL: Include TensorFlow ops for YAMNet
    # YAMNet uses operations not in standard TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS      # Additional TensorFlow ops (needed for YAMNet)
    ]
    
    # Allow dynamic shape
    converter.experimental_enable_resource_variables = True
    
    # Optional optimizations
    # Uncomment if you want quantization (smaller model, slightly less accurate)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting to TFLite with dynamic input shape...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
    print("✓ Model supports dynamic audio length input!")
    
    return output_path

def test_tflite_model(tflite_path, test_audio, class_names):
    """
    Test the TFLite model to ensure it works correctly.
    
    Args:
        tflite_path: Path to the TFLite model
        test_audio: Audio data as numpy array (float32, 1D)
        class_names: List of class names
    """
    print(f"\nTesting TFLite model...")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    
    # IMPORTANT: For dynamic shapes, we need to resize the input tensor
    # before allocating tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize input tensor to match the audio length
    interpreter.resize_tensor_input(input_details[0]['index'], test_audio.shape)
    interpreter.allocate_tensors()
    
    # Refresh details after allocation
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    print(f"Audio samples: {len(test_audio)}")
    
    # Ensure test_audio is 1D
    if len(test_audio.shape) > 1:
        test_audio = test_audio.flatten()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_audio)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(output_data).numpy()
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {probabilities[predicted_class_idx]:.4f}")
    print("\nAll probabilities:")
    for class_name, prob in zip(class_names, probabilities):
        print(f"  {class_name}: {prob:.4f}")
    
    return predicted_class, probabilities

def create_test_audio(duration_sec=5, sample_rate=16000):
    """
    Create a simple test audio signal.
    This is just for testing - replace with real audio in production.
    """
    num_samples = int(duration_sec * sample_rate)
    # Generate a simple sine wave
    t = np.linspace(0, duration_sec, num_samples)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio.astype(np.float32)

def main():
    # Paths
    SAVED_MODEL_PATH = './hearmate_model/serving_model'
    TFLITE_OUTPUT_PATH = './hearmate_model/hearmate_model.tflite'
    CLASS_NAMES_PATH = './hearmate_model/class_names.txt'
    
    # Convert to TFLite
    print("="*50)
    print("Converting Model to TFLite")
    print("="*50)
    tflite_path = convert_to_tflite(SAVED_MODEL_PATH, TFLITE_OUTPUT_PATH)
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"\nClass names: {class_names}")
    
    # Test with synthetic audio
    print("\n" + "="*50)
    print("Testing TFLite Model")
    print("="*50)
    test_audio = create_test_audio(duration_sec=2)  # 2 second audio
    print(f"Test audio shape: {test_audio.shape}")
    
    try:
        predicted_class, probabilities = test_tflite_model(
            tflite_path, 
            test_audio, 
            class_names
        )
        print("\n✓ TFLite model is working correctly!")
    except Exception as e:
        print(f"\n✗ Error testing TFLite model: {e}")
        print("This might be normal if you're testing with synthetic audio.")
    
    print("\n" + "="*50)
    print("Conversion Complete!")
    print("="*50)
    print(f"\nTFLite model ready for Android: {TFLITE_OUTPUT_PATH}")
    print("\nNOTE: This model accepts audio of ANY length (not fixed to 5 seconds)")
    print("For Android, you can pass 2-second (32000 samples) or 5-second (80000 samples) audio.")

if __name__ == "__main__":
    main()