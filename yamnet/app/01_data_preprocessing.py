import os
import json
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment
from pathlib import Path


class AudioSetPreprocessor:
    def __init__(self, raw_audio_path, output_path, target_sr=16000, chunk_duration=5.0, apply_augmentation=True):
        """
        Initialize AudioSet Preprocessor
        """
        self.raw_audio_path = Path(raw_audio_path)
        self.output_path = Path(output_path)
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.apply_augmentation = apply_augmentation

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # --- Utility Functions -----
    # ---------------------------
    def convert_to_wav(self, input_path):
        """Convert non-WAV audio to WAV at target sample rate"""
        input_ext = input_path.suffix.lower()
        output_path = input_path.with_suffix(".wav")

        try:
            if input_ext == ".mp3":
                audio = AudioSegment.from_mp3(input_path)
            elif input_ext in [".flac", ".ogg", ".m4a"]:
                audio = AudioSegment.from_file(input_path)
            else:
                return str(input_path)

            audio = audio.set_frame_rate(self.target_sr)
            audio = audio.set_channels(1)
            audio.export(output_path, format="wav")
            return str(output_path)
        except Exception as e:
            print(f"Error converting {input_path.name}: {e}")
            return None

    def normalize_audio(self, audio):
        max_val = np.abs(audio).max()
        return audio / max_val if max_val > 0 else audio

    def trim_silence(self, audio, sr, top_db=20):
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def create_chunks(self, audio, sr):
        """Create overlapping 5s chunks"""
        chunk_samples = int(self.chunk_duration * sr)
        chunks = []
        if len(audio) < chunk_samples:
            padded = np.pad(audio, (0, chunk_samples - len(audio)))
            chunks.append(padded)
        else:
            hop_length = chunk_samples // 2
            for i in range(0, len(audio) - chunk_samples + 1, hop_length):
                chunks.append(audio[i:i + chunk_samples])
        return chunks

    # ---------------------------
    # --- Augmentation ----------
    # ---------------------------
    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def time_shift(self, audio, sr, shift_max=0.2):
        shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
        return np.roll(audio, shift)

    def pitch_shift(self, audio, sr, n_steps=None):
        n_steps = n_steps if n_steps is not None else np.random.choice([-2, 2])
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    # ---------------------------
    # --- Processing Pipeline ---
    # ---------------------------
    def process_audio_file(self, input_path, output_dir):
        """Resample, normalize, trim, chunk, augment"""
        try:
            audio, sr = librosa.load(input_path, sr=self.target_sr)
            audio = self.normalize_audio(audio)
            audio = self.trim_silence(audio, sr)

            chunks = self.create_chunks(audio, sr)
            base_name = input_path.stem
            processed_files = []

            for i, chunk in enumerate(chunks):
                output_file = output_dir / f"{base_name}_chunk{i}.wav"
                sf.write(output_file, chunk, sr)
                processed_files.append(str(output_file))

            # Apply augmentation
            if self.apply_augmentation and len(chunks) > 0:
                chunk = chunks[0]

                noisy = self.add_noise(chunk)
                sf.write(output_dir / f"{base_name}_noisy.wav", noisy, sr)
                processed_files.append(str(output_dir / f"{base_name}_noisy.wav"))

                shifted = self.time_shift(chunk, sr)
                sf.write(output_dir / f"{base_name}_shifted.wav", shifted, sr)
                processed_files.append(str(output_dir / f"{base_name}_shifted.wav"))

                pitched = self.pitch_shift(chunk, sr)
                sf.write(output_dir / f"{base_name}_pitched.wav", pitched, sr)
                processed_files.append(str(output_dir / f"{base_name}_pitched.wav"))

            return processed_files
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            return []

    def run(self):
        print("=" * 60)
        print("AUDIOSET PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Input:  {self.raw_audio_path}")
        print(f"Output: {self.output_path}")
        print(f"Target SR: {self.target_sr} Hz")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Augmentation: {'ON' if self.apply_augmentation else 'OFF'}")
        print("=" * 60)

        dataset_info = {"classes": [], "files": {}, "statistics": {}}

        # Iterate over class folders
        for class_dir in sorted(self.raw_audio_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            dataset_info["classes"].append(class_name)
            print(f"\nProcessing class: {class_name}")

            output_dir = self.output_path / class_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all audio files
            audio_files = []
            for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                audio_files.extend(class_dir.glob(f"*{ext}"))

            dataset_info["files"][class_name] = []
            for file_path in tqdm(audio_files, desc=f"{class_name}"):
                if file_path.suffix.lower() != ".wav":
                    file_path = Path(self.convert_to_wav(file_path))
                    if not file_path:
                        continue

                processed_files = self.process_audio_file(file_path, output_dir)
                dataset_info["files"][class_name].extend(processed_files)

            dataset_info["statistics"][class_name] = {
                "original_files": len(audio_files),
                "processed_files": len(dataset_info["files"][class_name]),
            }

        # Save JSON summary
        info_path = self.output_path / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        for class_name in dataset_info["classes"]:
            stats = dataset_info["statistics"][class_name]
            ratio = stats["processed_files"] / max(stats["original_files"], 1)
            print(f"{class_name:35s}: {stats['original_files']:3d} → {stats['processed_files']:4d} ({ratio:.1f}x)")

        print("\nProcessed data saved to:", self.output_path)
        print("Dataset info JSON:", info_path)
        print("\n✅ PREPROCESSING COMPLETE")


# ---------------------------
# --- Run Script -----------
# ---------------------------
def main():
    raw_audio_path = r"E:\yamnet\data\raw"
    output_path = r"E:\yamnet\data\processed"

    preprocessor = AudioSetPreprocessor(
        raw_audio_path=raw_audio_path,
        output_path=output_path,
        target_sr=16000,
        chunk_duration=5.0,
        apply_augmentation=True
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
