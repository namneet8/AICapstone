# Approach 1: Audio Classification Pipeline Using YAMNet Embeddings

## Overview

**Approach 1** is a machine learning pipeline for classifying audio events into **five categories**:
**Alarm_Clock**, **Car_Horn**, **Glass_Breaking**, **Gunshot**, and **Siren**.

It leverages the **pre-trained YAMNet** model (from [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)) as a **frozen feature extractor** to generate **1024-dimensional embeddings** from preprocessed **0.96-second audio frames** (at 16 kHz, 15,360 samples per frame).

These embeddings are used to train and evaluate multiple classifiers.
Non-linear classifiers are **retrained as Keras MLPs** for compatibility with TensorFlow `SavedModel` format.
The final pipeline integrates YAMNet, a feature scaler, and the classifier into deployable models.

Model performance is evaluated on a **held-out test set**, reporting **accuracy, F1-score, precision, recall**, and **inference latency**.

---

## Dataset Summary

This approach processes **7,100 preprocessed audio frames** in total:

| Split      | Samples | Purpose               |
| :--------- | :-----: | :-------------------- |
| Training   |  ~4,970 | Model fitting         |
| Validation |   ~710  | Hyperparameter tuning |
| Test       |  ~1,420 | Final evaluation      |

---

## Directory Structure

```
../data/
├── processed/                     # Preprocessed audio frames & metadata
└── approach1/
    ├── features/                  # Extracted YAMNet embeddings
    │   ├── yamnet_features.npy
    │   ├── yamnet_labels.npy
    │   └── yamnet_features_metadata.csv
    ├── train_set/                 # Training data
    │   ├── X_train.npy
    │   └── y_train.npy
    ├── test_set/                  # Test data
    │   ├── X_test_raw_audio.npy
    │   ├── y_test.npy
    │   └── test_frame_paths.npy
```

```
../models/models_approach1/        # Trained classifiers, scaler, label encoder, SavedModels
../results/results_approach1/      # Model metrics, plots, comparison tables
```

---

## Dependencies

* **Python** 3.12+
* **Libraries:**
  `numpy`, `pandas`, `tensorflow (2.15+)`, `tensorflow_hub`,
  `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `seaborn`, `tqdm`

> No extra dependencies beyond standard scientific Python stack.

---

## Pipeline Steps

### **1️ Feature Extraction — `01_feature_extraction.py`**

* Load preprocessed frames from `../data/processed/`
* Use frozen **YAMNet** to extract **averaged 1024-dim embeddings**
* Save features, labels, and metadata
* (Optional) Visualize embeddings with **PCA** or **t-SNE**

---

### **2️ Classifier Training & Evaluation — `02_classifier_training_evaluation.py`**

* Load YAMNet features and labels
* Split dataset:

  * 80% train+val → (87.5% train / 12.5% val)
  * 20% held-out test
* Scale features with **StandardScaler**
* Train **5 classifiers** (with 3-fold GridSearchCV):

| Classifier          | Type              | Notes                      |
| :------------------ | :---------------- | :------------------------- |
| Logistic Regression | Linear            | Baseline                   |
| Random Forest       | Non-linear        | Tree-based                 |
| XGBoost             | Gradient Boosting | Strongest performer        |
| SVM                 | Kernel-based      | Non-linear separation      |
| MLP                 | Neural Network    | Tuned for TF compatibility |

* Evaluate with:

  * Accuracy, F1, Precision, Recall, ROC-AUC
  * Confusion Matrix and ROC plots
* Save trained models (`.pkl`), scaler, label encoder, and CSV reports.

---

### **3️ Full Pipeline Assembly — `03_build_full_pipeline.py`**

* Load best classifiers + scaler + label encoder
* Retrain **non-LR classifiers** as equivalent **Keras MLPs**
* Construct full TensorFlow pipelines:
  `Audio Input → YAMNet → Scaler → Classifier`
* Export to **TensorFlow SavedModel** format
* Generate **inventory CSV** with model sizes and validation metrics

---

### **4️ Final Test Evaluation — `04_test_eval.py`**

* Load SavedModels and **reconstructed raw test audio**
* Run inference and compute:

  * Accuracy, F1-score, Precision, Recall
  * Confusion Matrix, ROC-AUC
  * **Latency** (ms/sample)
* Save all metrics, confusion matrices, and ROC curves to results directory.

---

## Key Components

| Component                  | Description                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| **YAMNet**                 | Pre-trained on Google AudioSet (521 classes), used for embedding extraction only (frozen). |
| **Feature Dimensionality** | 1024 per 0.96s frame                                                                       |
| **MLP Conversion**         | Non-linear models retrained as small Keras MLPs for end-to-end TF export.                  |
| **Model Size**             | ~3.5–3.6 MB each                                                                           |

---

## Performance Summary (Validation F1)

| Rank | Classifier          |  F1 Score  |
| :--: | :------------------ | :--------: |
|  1   | XGBoost             | **0.9257** |
|  2   | MLP                 | **0.9250** |
|  3   | Random Forest       | **0.9213** |
|  4️   | SVM                 | **0.9209** |
|  5️   | Logistic Regression | **0.9191** |

---

## Test Evaluation

* **Held-out set:** 1,420 samples
* **Raw audio reconstructed** for full pipeline testing
* Reports: accuracy, precision, recall, F1, latency, confusion matrix, and ROC-AUC.

---

## Usage

Ensure preprocessed data is available in `../data/processed`
(from a prior preprocessing script, not included in this repo).

Then, execute the scripts sequentially:

```bash
python 01_feature_extraction.py
python 02_classifier_training_evaluation.py
python 03_build_full_pipeline.py
python 04_test_eval.py
```

View results in:

```
../results/results_approach1/
```

---

## Outputs

* **Model Comparison:** `model_comparison.csv`
* **Full Model Inventory:** `full_model_inventory.csv`
* **SavedModels:** TensorFlow exportable end-to-end models
* **Evaluation Reports:** Confusion matrices, ROC curves, metric summaries


