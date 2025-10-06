"""
Step 2: Extract YAMNet Embeddings
---------------------------------
Extracts embeddings from processed audio files using YAMNet
and visualizes embedding distributions and t-SNE projections.
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams.update({'font.size': 12})

class YAMNetEmbeddingExtractor:
    def __init__(self, processed_dir, output_dir):
        """
        Args:
            processed_dir: Path to processed dataset (organized by class)
            output_dir: Directory to save embeddings and metadata
        """
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = 16000
        self.model_handle = 'https://tfhub.dev/google/yamnet/1'

        # Load dataset info if available
        dataset_info_path = self.processed_dir / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, "r") as f:
                info = json.load(f)
                self.class_names = info["classes"]
        else:
            # fallback: read folder names
            self.class_names = sorted(
                [p.name for p in self.processed_dir.iterdir() if p.is_dir()]
            )

        print(f"\nDetected classes: {self.class_names}")

        # Load YAMNet model
        print("\nLoading YAMNet model from TensorFlow Hub...")
        self.model = hub.load(self.model_handle)
        print("✓ YAMNet model loaded successfully!")
        print("Model output embedding dimension: 1024")

    def load_audio(self, filepath):
        """Load audio file in float32 mono at 16kHz"""
        wav, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        wav = wav.astype(np.float32)
        return wav

    def extract_embedding(self, wav):
        """Run YAMNet and return averaged embedding"""
        try:
            _, embeddings, _ = self.model(wav)
            avg_embedding = np.mean(embeddings.numpy(), axis=0)
            return avg_embedding
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def run(self):
        """Extract embeddings for the entire dataset"""
        all_embeddings, all_labels, all_filenames = [], [], []

        print("\n" + "=" * 60)
        print("EXTRACTING YAMNET EMBEDDINGS")
        print("=" * 60)

        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.processed_dir / class_name
            if not class_path.exists():
                print(f"⚠️  Skipping missing class folder: {class_name}")
                continue

            files = list(class_path.glob("*.wav"))
            print(f"\n{class_name}: {len(files)} files")

            for f in tqdm(files, desc=f"  Extracting {class_name}"):
                wav = self.load_audio(f)
                emb = self.extract_embedding(wav)
                if emb is not None:
                    all_embeddings.append(emb)
                    all_labels.append(class_idx)
                    all_filenames.append(f.name)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        np.save(self.embeddings_dir / "embeddings.npy", embeddings)
        np.save(self.embeddings_dir / "labels.npy", labels)

        metadata = {
            "class_names": self.class_names,
            "filenames": all_filenames,
            "num_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "samples_per_class": {
                c: int(np.sum(labels == i)) for i, c in enumerate(self.class_names)
            },
        }

        with open(self.embeddings_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n✓ Embeddings extraction complete!")
        print(f"Saved to {self.embeddings_dir}")
        print(f"Total samples: {metadata['num_samples']}")
        print(f"Embedding dimension: {metadata['embedding_dim']}")

        self.visualize_embeddings(embeddings, labels)

    # -----------------------------------------------------------
    # Visualization Utilities
    # -----------------------------------------------------------

    def visualize_embeddings(self, embeddings, labels):
        """Run t-SNE and visualize embedding distributions"""
        if len(embeddings) == 0:
            print("⚠️ No embeddings to visualize.")
            return

        print("\nComputing t-SNE (2D projection)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 7))
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                label=class_name,
                s=40,
                alpha=0.7,
            )
        plt.legend()
        plt.title("t-SNE Visualization of YAMNet Embeddings", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot mean and std per class
        self.plot_embedding_stats(embeddings, labels)

    def plot_embedding_stats(self, embeddings, labels):
        """Plot mean and std deviation per class"""
        class_means, class_stds = [], []
        for i in range(len(self.class_names)):
            class_embeddings = embeddings[labels == i]
            class_means.append(np.mean(class_embeddings))
            class_stds.append(np.std(class_embeddings))

        x = np.arange(len(self.class_names))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(x, class_means, alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45)
        axes[0].set_title("Mean Embedding Value per Class")

        axes[1].bar(x, class_stds, color="orange", alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45)
        axes[1].set_title("Embedding Std Dev per Class")

        plt.tight_layout()
        plt.show()


def main():
    processed_dir = r"E:/yamnet/data/processed"
    output_dir = r"E:/yamnet/data"

    extractor = YAMNetEmbeddingExtractor(processed_dir, output_dir)
    extractor.run()


if __name__ == "__main__":
    main()
