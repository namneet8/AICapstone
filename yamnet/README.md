# Audio Classification Pipeline - Setup Instructions

## Overview

This project classifies audio into 5 categories using YAMNet embeddings and a custom trained neural network:

- Alarm clock
- Explosion
- Gunshot, gunfire
- Siren
- Vehicle horn, car horn, honking

## Prerequisites

### 1. Install Python (3.8 or higher)

Download from: https://www.python.org/downloads/

### 2. Install Required Libraries

Open Command Prompt or Terminal and run:

```bash
pip install tensorflow tensorflow-hub numpy librosa soundfile scikit-learn pyaudio tqdm
```

**Note for Windows PyAudio installation:**
If PyAudio fails to install, download the wheel file:

1. Go to: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Download the appropriate `.whl` file for your Python version
3. Install: `pip install PyAudio-0.2.11-cp3X-cp3X-win_amd64.whl`

## Project Structure

```
yamnet/
├── data/
│   ├── raw/                    # Raw audio files (if any)
│   ├── processed/              # Processed audio files by class
│   │   ├── Alarm clock/
│   │   ├── Explosion/
│   │   ├── Gunshot, gunfire/
│   │   ├── Siren/
│   │   └── Vehicle horn, car horn, honking/
│   └── embeddings/             # YAMNet embeddings
│       ├── embeddings.npy
│       ├── labels.npy
│       └── metadata.json
├── models/
│   ├── finetuned/              # Trained models
│   │   ├── best_model.h5
│   │   ├── final_model.h5
│   │   ├── model_metadata.json
│   │   └── model_history.json
│   └── tflite/                 # TFLite models for deployment
│       ├── audio_classifier_float16.tflite
│       ├── audio_classifier_float32.tflite
│       ├── audio_classifier_int8.tflite
│       └── conversion_metadata.json
└── app/
    ├── 01_data_preprocessing.py
    ├── 02_embedding_extraction.py
    ├── 03_model_training.py
    ├── 04_model_evaluation.py
    ├── 05_tflite_conversion.py
    ├── 06_realtime_classifier.py
    └── master_pipeline.py
```

## Step-by-Step Setup

### Step 1: Organize Your Audio Files

Place your audio files in:

```
E:\GoogleAudioSet\raw_audio\segmented_audio\
```

**Audio File Naming Convention:**
Name your files with keywords to help automatic classification:

- `alarm_001.wav`, `clock_sound.wav` → Alarm clock
- `explosion_blast.wav`, `boom_001.wav` → Explosion
- `gunshot_01.wav`, `gunfire.wav` → Gunshot, gunfire
- `siren_police.wav`, `ambulance.wav` → Siren
- `car_horn.wav`, `honking.wav` → Vehicle horn

**Supported Formats:** WAV, MP3, FLAC, OGG, M4A

### Step 2: Update Paths in Scripts

Edit the following paths in your scripts to match your setup:

**In `master_pipeline.py`:**

```python
RAW_AUDIO_PATH = r"E:/.../raw"
BASE_OUTPUT_PATH = r"E:/.../yamnet"
```

**In `realtime_classifier.py`:**

```python
tflite_model_path = r'E:/.../audio_classifier_float32.tflite'
```

### Step 3: Run the Pipeline

#### Option A: Run Complete Pipeline (Recommended)

```bash
python master_pipeline.py
```

This runs all steps automatically:

1. Data preprocessing
2. Embedding extraction
3. Model training
4. TFLite conversion

#### Option B: Run Steps Individually

**Step 1: Preprocess Audio**

```bash
python 01_data_preprocessing.py
```

- Resamples audio to 16kHz
- Normalizes audio
- Organizes by class

**Step 2: Extract Embeddings**

```bash
python 02_embedding_extraction.py
```

- Downloads YAMNet model
- Extracts 1024-dimensional embeddings
- Saves embeddings and labels

**Step 3: Train Model**

```bash
python 03_model_training.py
```

- Trains neural network classifier
- Uses early stopping and learning rate reduction
- Saves best model

**Step 4: Evaluate Model**

```bash
python 04_model_evaluation.py
```

- Evaluate the trained audio classification model on the test set
- Generates confidence plots, ROC/PR curves, misclassification analysis
- inference speed measurement

**Step 5: Convert to TFLite**

```bash
python 05_tflite_conversion.py
```

- Converts to TFLite format
- Creates both float32 and quantized versions
- Verifies model works

**Step 6: Run Real-time Classifier**

```bash
python 06_realtime_classifier.py
```

- Opens GUI application
- Listens to microphone
- Displays real-time predictions

## Troubleshooting

### Issue: No audio files found

**Solution:**

- Check that audio files are in the correct directory
- Verify file extensions are supported (.wav, .mp3, etc.)
- Ensure files are named with appropriate keywords

### Issue: YAMNet download fails

**Solution:**

- Check internet connection
- If behind proxy, configure: `export HTTP_PROXY=http://proxy:port`
- Try downloading manually from: https://tfhub.dev/google/yamnet/1

### Issue: PyAudio not working

**Solution:**

- Windows: Download wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Mac: `brew install portaudio && pip install pyaudio`
- Linux: `sudo apt-get install portaudio19-dev && pip install pyaudio`

### Issue: Microphone not detected

**Solution:**

- Check microphone is connected and enabled
- Grant microphone permissions to Python
- Close other applications using microphone
- Try different microphone in PyAudio device list

### Issue: Low accuracy

**Solution:**

- Collect more training data per class (aim for 50+ samples each)
- Ensure audio files are correctly classified
- Try adjusting confidence threshold in realtime_classifier.py
- Retrain model with more epochs

### Issue: Out of memory during training

**Solution:**

- Reduce batch size in training script (try 16 or 8)
- Close other applications
- Use smaller model architecture

## Tips for Best Results

1. **Quality Audio Data:**

   - Use clear, unambiguous audio samples
   - Aim for 50-100 samples per class minimum
   - Ensure good variety within each class

2. **File Naming:**

   - Use clear keywords in filenames
   - Keep filenames descriptive
   - Avoid special characters

3. **Training:**

   - Monitor validation accuracy
   - If overfitting, increase dropout rates
   - If underfitting, add more layers or neurons

4. **Real-time Classification:**
   - Adjust confidence_threshold (0.5-0.7 works well)
   - Test in quiet environment first
   - Speak clearly near microphone

## Expected Timeline

- **Preprocessing:** 1-5 minutes (depends on number of files)
- **Embedding Extraction:** 5-20 minutes (first run downloads YAMNet)
- **Training:** 5-15 minutes (depends on data size)
- **TFLite Conversion:** 1-2 minutes
- **Total:** ~15-45 minutes for complete pipeline

## Performance Benchmarks

With 50+ samples per class:

- **Expected Accuracy:** 70-90%
- **Real-time Latency:** <500ms
- **Model Size:** ~2-5 MB (float32), ~500KB-1MB (quantized)

## Next Steps

After successful setup:

1. Test real-time classifier with various sounds
2. Collect more data for classes with low accuracy
3. Experiment with different confidence thresholds
4. Consider adding more sound classes
5. Deploy to mobile/edge devices using TFLite

## Support

For issues or questions:

1. Check error messages carefully
2. Verify all paths are correct
3. Ensure all requirements are installed
4. Check that audio files exist and are readable

## License & Credits

- YAMNet model: Google Research (Apache 2.0)
- TensorFlow: Apache 2.0
- This implementation: Educational use
