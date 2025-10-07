"""
Step 2: Extract YAMNet Embeddings
---------------------------------
Extracts embeddings from processed audio files using YAMNet
and visualizes embedding distributions, t-SNE projections, and transfer learning architecture.
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
import matplotlib.patches as mpatches
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

        # Visualizations
        self.visualize_transfer_learning_architecture()
        self.visualize_embeddings(embeddings, labels)

    # -----------------------------------------------------------
    # NEW: Transfer Learning Architecture Diagram
    # -----------------------------------------------------------

    def visualize_transfer_learning_architecture(self):
        """
        Visualize the transfer learning pipeline architecture
        Shows: Audio Input → YAMNet (Frozen) → Embeddings → Classifier (Trainable)
        """
        print("\nGenerating transfer learning architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Title
        ax.text(7, 7.2, 'Transfer Learning Architecture',
                ha='center', va='center', fontsize=18, weight='bold')
        ax.text(7, 6.7, 'Feature Extraction with Pre-trained YAMNet',
                ha='center', va='center', fontsize=12, style='italic', color='#666')

        # Audio Input
        input_box = mpatches.FancyBboxPatch((0.5, 3), 2, 2,
                                           boxstyle="round,pad=0.15",
                                           edgecolor='#2196F3',
                                           facecolor='#E3F2FD',
                                           linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.5, 4.5, '🎵', ha='center', va='center', fontsize=40)
        ax.text(1.5, 3.7, 'Audio Input', ha='center', va='center', 
                fontsize=11, weight='bold')
        ax.text(1.5, 3.3, '16kHz\n5 seconds', ha='center', va='center', 
                fontsize=9, color='#666')

        # Arrow 1
        ax.annotate('', xy=(3.5, 4), xytext=(2.7, 4),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

        # YAMNet (Pre-trained, Frozen)
        yamnet_box = mpatches.FancyBboxPatch((3.5, 2.5), 3, 3,
                                            boxstyle="round,pad=0.15",
                                            edgecolor='#FF9800',
                                            facecolor='#FFF3E0',
                                            linewidth=2)
        ax.add_patch(yamnet_box)
        ax.text(5, 5, 'YAMNet', ha='center', va='center', 
                fontsize=14, weight='bold', color='#E65100')
        ax.text(5, 4.5, '(Pre-trained)', ha='center', va='center', 
                fontsize=10, style='italic', color='#F57C00')
        
        # Frozen indicator
        frozen_badge = mpatches.FancyBboxPatch((4.2, 3.8), 1.6, 0.5,
                                              boxstyle="round,pad=0.05",
                                              edgecolor='#D32F2F',
                                              facecolor='#FFCDD2',
                                              linewidth=1.5)
        ax.add_patch(frozen_badge)
        ax.text(5, 4.05, '❄️ FROZEN', ha='center', va='center', 
                fontsize=9, weight='bold', color='#D32F2F')
        
        ax.text(5, 3.3, '2M AudioSet videos', ha='center', va='center', 
                fontsize=9, color='#666')
        ax.text(5, 2.9, '527 classes', ha='center', va='center', 
                fontsize=9, color='#666')

        # Arrow 2
        ax.annotate('', xy=(7.5, 4), xytext=(6.7, 4),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

        # Embeddings
        embed_box = mpatches.FancyBboxPatch((7.5, 2.8), 2.5, 2.4,
                                           boxstyle="round,pad=0.15",
                                           edgecolor='#4CAF50',
                                           facecolor='#E8F5E9',
                                           linewidth=2)
        ax.add_patch(embed_box)
        ax.text(8.75, 4.7, 'Embeddings', ha='center', va='center', 
                fontsize=12, weight='bold', color='#2E7D32')
        ax.text(8.75, 4.2, '1024-D', ha='center', va='center', 
                fontsize=11, weight='bold', color='#388E3C')
        ax.text(8.75, 3.7, 'Feature', ha='center', va='center', 
                fontsize=9, color='#666')
        ax.text(8.75, 3.4, 'Vectors', ha='center', va='center', 
                fontsize=9, color='#666')

        # Arrow 3
        ax.annotate('', xy=(11, 4), xytext=(10.2, 4),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

        # Custom Classifier (Trainable)
        classifier_box = mpatches.FancyBboxPatch((11, 2.5), 2.5, 3,
                                                boxstyle="round,pad=0.15",
                                                edgecolor='#9C27B0',
                                                facecolor='#F3E5F5',
                                                linewidth=2)
        ax.add_patch(classifier_box)
        ax.text(12.25, 5, 'Classifier', ha='center', va='center', 
                fontsize=13, weight='bold', color='#6A1B9A')
        
        # Trainable indicator
        train_badge = mpatches.FancyBboxPatch((11.45, 4.3), 1.6, 0.5,
                                             boxstyle="round,pad=0.05",
                                             edgecolor='#1976D2',
                                             facecolor='#BBDEFB',
                                             linewidth=1.5)
        ax.add_patch(train_badge)
        ax.text(12.25, 4.55, '🔥 TRAINABLE', ha='center', va='center', 
                fontsize=8, weight='bold', color='#0D47A1')
        
        ax.text(12.25, 3.7, 'Dense Layers', ha='center', va='center', 
                fontsize=9, color='#666')
        ax.text(12.25, 3.3, f'{len(self.class_names)} Classes', ha='center', va='center', 
                fontsize=10, weight='bold', color='#7B1FA2')
        ax.text(12.25, 2.9, 'Trainable custom classes', ha='center', va='center', 
                fontsize=9, style='italic', color='#666')

        # Legend/Information box
        info_box = mpatches.FancyBboxPatch((0.5, 0.3), 13, 1.5,
                                          boxstyle="round,pad=0.1",
                                          edgecolor='#757575',
                                          facecolor='#FAFAFA',
                                          linewidth=1.5,
                                          linestyle='--')
        ax.add_patch(info_box)
        
        ax.text(1, 1.5, '💡 Transfer Learning Strategy:', 
                ha='left', va='center', fontsize=11, weight='bold')
        ax.text(1, 1.1, '• YAMNet: Pre-trained on AudioSet (frozen weights)', 
                ha='left', va='center', fontsize=9)
        ax.text(1, 0.8, '• Embeddings: Rich 1024D audio features extracted automatically', 
                ha='left', va='center', fontsize=9)
        ax.text(1, 0.5, '• Classifier: Only train final layers for your specific task', 
                ha='left', va='center', fontsize=9)
        
        ax.text(13, 1.1, f'Dataset: {len(self.class_names)} classes', 
                ha='right', va='center', fontsize=10, weight='bold', 
                bbox=dict(boxstyle='round', facecolor='#E0E0E0', alpha=0.8))

        plt.tight_layout()
        
        # Save the figure
        save_path = self.embeddings_dir / "transfer_learning_architecture.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Architecture diagram saved to: {save_path}")
        
        plt.show()

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

        # Create color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                label=class_name,
                s=60,
                alpha=0.7,
                color=colors[i],
                edgecolors='white',
                linewidth=0.5
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.title("t-SNE Visualization of YAMNet Embeddings", 
                 fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save
        save_path = self.embeddings_dir / "tsne_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE visualization saved to: {save_path}")
        
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Mean values
        bars1 = axes[0].bar(x, class_means, alpha=0.8, color='steelblue', 
                           edgecolor='navy', linewidth=1.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].set_title("Mean Embedding Value per Class", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Mean Value", fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)

        # Standard deviation
        bars2 = axes[1].bar(x, class_stds, alpha=0.8, color='coral', 
                           edgecolor='darkred', linewidth=1.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].set_title("Embedding Std Dev per Class", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Standard Deviation", fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        
        # Save
        save_path = self.embeddings_dir / "embedding_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Embedding statistics saved to: {save_path}")
        
        plt.show()


def main():
    processed_dir = r"E:/yamnet/data/processed"
    output_dir = r"E:/yamnet/data"

    extractor = YAMNetEmbeddingExtractor(processed_dir, output_dir)
    extractor.run()


if __name__ == "__main__":
    main()