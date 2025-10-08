"""
Embed Metadata Directly into TFLite Model
==========================================
This script embeds metadata directly into the TFLite file using the TFLite schema.
The metadata will be part of the model file itself.

Requirements:
    pip install flatbuffers tensorflow
"""

import json
import zipfile
import io
from pathlib import Path
import tensorflow as tf
from tensorflow import lite
from tensorflow.lite.python import schema_py_generated as schema_fb
import flatbuffers


class TFLiteMetadataEmbedder:
    def __init__(self, tflite_dir):
        """
        Args:
            tflite_dir: Directory containing TFLite models and metadata
        """
        self.tflite_dir = Path(tflite_dir)
        
    def create_metadata_buffer(self, class_names, model_info):
        """
        Create metadata buffer using FlatBuffers
        """
        builder = flatbuffers.Builder(1024)
        
        # Create metadata JSON string
        metadata_json = {
            "name": "Audio Classifier",
            "description": "YAMNet-based audio event classification",
            "version": "1.0.0",
            "author": "Your Name",
            "license": "Apache-2.0",
            "min_parser_version": "1.0.0",
            "subgraph_metadata": [{
                "input_tensor_metadata": [{
                    "name": "yamnet_embedding",
                    "description": "1024-dimensional YAMNet embedding vector",
                    "content": {
                        "content_properties_type": "FeatureProperties",
                        "content_properties": {}
                    },
                    "stats": {}
                }],
                "output_tensor_metadata": [{
                    "name": "probability",
                    "description": "Probability of each class",
                    "content": {
                        "content_properties_type": "FeatureProperties",
                        "content_properties": {}
                    },
                    "stats": {},
                    "associated_files": [{
                        "name": "labels.txt",
                        "description": "Class labels",
                        "type": "TENSOR_AXIS_LABELS"
                    }]
                }]
            }],
            "model_info": model_info
        }
        
        # Convert to string
        metadata_str = json.dumps(metadata_json, indent=2)
        metadata_offset = builder.CreateString(metadata_str)
        
        # Create associated file for labels
        labels_content = "\n".join(class_names).encode('utf-8')
        labels_offset = builder.CreateByteVector(labels_content)
        
        return metadata_str, labels_content
    
    def embed_metadata_method1_flatbuffers(self, model_path, class_names, model_info):
        """
        Method 1: Using FlatBuffers to properly embed metadata
        This is the most compatible approach
        """
        print(f"\nEmbedding metadata in {model_path.name}...")
        
        # Read the TFLite model
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        # Parse the model
        model = schema_fb.Model.GetRootAsModel(model_content, 0)
        
        # Create new model with metadata
        builder = flatbuffers.Builder(len(model_content) + 8192)
        
        # Create metadata JSON
        metadata_json, labels_content = self.create_metadata_buffer(class_names, model_info)
        
        # Create metadata buffer vector
        metadata_name = builder.CreateString("TFLITE_METADATA")
        metadata_buffer = builder.CreateByteVector(metadata_json.encode('utf-8'))
        
        # Create labels file buffer
        labels_name = builder.CreateString("labels.txt")
        labels_buffer = builder.CreateByteVector(labels_content)
        
        # Start building metadata
        schema_fb.BufferStart(builder)
        schema_fb.BufferAddData(builder, metadata_buffer)
        metadata_buffer_offset = schema_fb.BufferEnd(builder)
        
        schema_fb.BufferStart(builder)
        schema_fb.BufferAddData(builder, labels_buffer)
        labels_buffer_offset = schema_fb.BufferEnd(builder)
        
        # Copy original model structure and add metadata
        # This requires rebuilding the entire model structure
        # (Complex FlatBuffers manipulation)
        
        print("⚠️  Method 1 requires complex FlatBuffers manipulation")
        print("    Using Method 2 instead (ZIP approach)...")
        
        return self.embed_metadata_method2_zip(model_path, class_names, model_info)
    
    def embed_metadata_method2_zip(self, model_path, class_names, model_info):
        """
        Method 2: Create a ZIP archive containing model + metadata
        This is TensorFlow Lite's recommended approach for including associated files
        """
        output_path = model_path.with_name(model_path.stem + "_with_metadata.tflite")
        
        # Read original model
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        # Create metadata JSON
        metadata_json = {
            "name": "Audio Classifier",
            "description": "YAMNet-based audio event classification",
            "version": "1.0.0",
            "author": "Your Name",
            "license": "Apache-2.0",
            "classes": class_names,
            "input": {
                "name": "yamnet_embedding",
                "shape": [1, 1024],
                "dtype": "float32",
                "description": "YAMNet embedding vector (1024-dimensional)"
            },
            "output": {
                "name": "probabilities",
                "shape": [1, len(class_names)],
                "dtype": "float32",
                "description": "Class probabilities"
            },
            "preprocessing": {
                "sample_rate": 16000,
                "chunk_duration_seconds": 5.0,
                "normalization": "per_sample_max",
                "silence_trimming_db": 20
            },
            "performance": model_info
        }
        
        # Create labels.txt
        labels_content = "\n".join(class_names)
        
        # Write model with embedded files using TFLite's metadata approach
        # Actually, we'll use a simpler method that's compatible with TFLite
        
        # The proper way: Use TensorFlow Lite's metadata writer
        # But since tflite_support is deprecated, we'll create companion files
        
        # Save metadata as JSON companion file
        metadata_path = model_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        
        # Save labels file
        labels_path = model_path.with_suffix('.labels.txt')
        with open(labels_path, 'w') as f:
            f.write(labels_content)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        print(f"✓ Labels saved to: {labels_path}")
        
        # Also create a ZIP bundle for easy distribution
        zip_path = model_path.with_suffix('.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, model_path.name)
            zf.writestr('metadata.json', json.dumps(metadata_json, indent=2))
            zf.writestr('labels.txt', labels_content)
            
            # Add usage instructions
            readme = f"""# Audio Classifier Model Package
            
Model: {model_path.name}
Classes: {len(class_names)}
Input: 1024-dimensional YAMNet embedding (float32)
Output: {len(class_names)} class probabilities (float32)

## Usage
1. Load the .tflite model file
2. Extract YAMNet embeddings from audio (16kHz, 5-second chunks)
3. Feed embeddings to this classifier
4. Apply softmax to get probabilities

## Classes
{chr(10).join(f"{i}. {name}" for i, name in enumerate(class_names))}

## Performance
- Accuracy: {model_info.get('accuracy', 'N/A')}
- Size: {model_info.get('size_kb', 'N/A')} KB
- Inference time: {model_info.get('time_ms', 'N/A')} ms
"""
            zf.writestr('README.txt', readme)
        
        print(f"✓ ZIP bundle created: {zip_path}")
        
        return metadata_path, labels_path, zip_path
    
    def embed_metadata_method3_tensorflow(self, model_path, class_names, model_info):
        """
        Method 3: Use TensorFlow's built-in metadata populator
        This actually embeds data INTO the .tflite file
        """
        try:
            from tensorflow.lite.python import metadata as tflite_metadata
            from tensorflow.lite.python.metadata_writers import metadata_info as metadata_info_lib
            from tensorflow.lite.python.metadata_writers import writer_utils
        except ImportError:
            print("⚠️  TensorFlow metadata tools not available")
            print("   Falling back to companion files method...")
            return self.embed_metadata_method2_zip(model_path, class_names, model_info)
        
        output_path = model_path.with_name(model_path.stem + "_with_metadata.tflite")
        
        # Create labels file temporarily
        labels_path = self.tflite_dir / "temp_labels.txt"
        with open(labels_path, 'w') as f:
            f.write("\n".join(class_names))
        
        try:
            # Read model
            with open(model_path, 'rb') as f:
                model_buffer = f.read()
            
            # Create metadata
            model_meta = tflite_metadata.ModelMetadataT()
            model_meta.name = "Audio Classifier"
            model_meta.description = "YAMNet-based audio event classification model"
            model_meta.version = "1.0.0"
            model_meta.author = "Your Name"
            model_meta.license = "Apache-2.0"
            
            # Input metadata
            input_meta = tflite_metadata.TensorMetadataT()
            input_meta.name = "yamnet_embedding"
            input_meta.description = "1024-dimensional YAMNet embedding"
            input_meta.content = tflite_metadata.ContentT()
            input_meta.content.contentPropertiesType = tflite_metadata.ContentProperties.FeatureProperties
            
            # Output metadata with labels
            output_meta = tflite_metadata.TensorMetadataT()
            output_meta.name = "probability"
            output_meta.description = "Probabilities for each class"
            output_meta.content = tflite_metadata.ContentT()
            output_meta.content.contentPropertiesType = tflite_metadata.ContentProperties.FeatureProperties
            
            # Associate labels file
            output_meta.associatedFiles = [metadata_info_lib.LabelFileMd(file_path=str(labels_path))]
            
            # Create subgraph
            subgraph = tflite_metadata.SubGraphMetadataT()
            subgraph.inputTensorMetadata = [input_meta]
            subgraph.outputTensorMetadata = [output_meta]
            
            model_meta.subgraphMetadata = [subgraph]
            
            # Create populated buffer
            b = flatbuffers.Builder(0)
            b.Finish(model_meta.Pack(b), tflite_metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
            metadata_buf = b.Output()
            
            # Populate metadata
            populator = tflite_metadata.MetadataPopulator.with_model_buffer(model_buffer)
            populator.load_metadata_buffer(metadata_buf)
            populator.load_associated_files([str(labels_path)])
            populator.populate()
            
            # Write output
            with open(output_path, 'wb') as f:
                f.write(populator.get_model_buffer())
            
            print(f"✓ Metadata embedded in: {output_path}")
            
            # Cleanup
            labels_path.unlink()
            
            return output_path
            
        except Exception as e:
            print(f"⚠️  Error embedding metadata: {e}")
            print("   Falling back to companion files method...")
            if labels_path.exists():
                labels_path.unlink()
            return self.embed_metadata_method2_zip(model_path, class_names, model_info)
    
    def run(self, use_method=3):
        """
        Add metadata to all TFLite models
        
        Args:
            use_method: 1=FlatBuffers, 2=ZIP bundle, 3=TensorFlow built-in (recommended)
        """
        print("\n" + "="*60)
        print("EMBEDDING METADATA IN TFLITE MODELS")
        print("="*60)
        
        # Load conversion metadata
        metadata_file = self.tflite_dir / "conversion_metadata.json"
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            print("   Please run 05_tflite_conversion.py first")
            return
        
        with open(metadata_file, 'r') as f:
            conv_meta = json.load(f)
        
        class_names = conv_meta["class_names"]
        print(f"Classes: {class_names}")
        
        # Process each model
        for model_type in ["float32", "float16", "int8"]:
            model_info = conv_meta["tflite_models"][model_type]
            model_path = Path(model_info["path"])
            
            if not model_path.exists():
                print(f"⚠️  Skipping missing model: {model_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {model_type.upper()}")
            print(f"{'='*60}")
            
            if use_method == 1:
                result = self.embed_metadata_method1_flatbuffers(
                    model_path, class_names, model_info
                )
            elif use_method == 2:
                result = self.embed_metadata_method2_zip(
                    model_path, class_names, model_info
                )
            else:  # method 3
                result = self.embed_metadata_method3_tensorflow(
                    model_path, class_names, model_info
                )
        
        print("\n" + "="*60)
        print("✓ METADATA EMBEDDING COMPLETE!")
        print("="*60)
        print("\nDeployment options:")
        print("1. Use *_with_metadata.tflite files (if Method 3 worked)")
        print("2. Use .zip bundles that include model + metadata")
        print("3. Use original .tflite with companion .metadata.json files")


def main():
    embedder = TFLiteMetadataEmbedder(
        tflite_dir=r"E:/yamnet/models/tflite"
    )
    
    # Try method 3 (TensorFlow built-in) first
    # Falls back to method 2 (ZIP bundles) if not available
    embedder.run(use_method=3)


if __name__ == "__main__":
    main()