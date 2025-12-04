import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import random

# ======================================================
# Constants
# ======================================================
TARGET_SR = 16000
BATCH_SIZE = 32
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ======================================================
# Load and resample audio
# ======================================================
audio_path = r"D:\centennial\centennial 2025 fall\comp385\assignment\phase1\dataset\GoogleAudioSet\raw_audio\segmented_audio\Vehicle horn, car horn, honking\2X1DAn1AhlU_0.wav"  

y, sr = librosa.load(audio_path, sr=TARGET_SR)

print(f"Loaded audio length: {len(y)}, Sample Rate: {sr}")

# ======================================================
# ----- 1. Plot waveform (NEW API) -----
# ======================================================
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()

plt.savefig("Amplitude.png")  
plt.show()

plt.close()
print("Saved: Amplitude.png")

# ======================================================
# ----- 2. Create STFT spectrogram -----
# ======================================================
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# ======================================================
# ----- 3. Plot spectrogram -----
# ======================================================
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.tight_layout()
plt.show()

# ======================================================
# Save processed WAV file
# ======================================================
sf.write("output_resampled.wav", y, sr)
print("Saved: output_resampled.wav")

# ======================================================
# Save spectrogram image
# ======================================================
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar()
plt.title("Spectrogram Image")
plt.tight_layout()
plt.savefig("spectrogram.png")
plt.close()
print("Saved: spectrogram.png")


