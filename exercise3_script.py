import os
import string
from PIL import Image
import numpy as np
import librosa

# ---- TEXT -----
def load_text_file(filepath):
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.lower()
            if line:
                lines.append(line)
    return lines

def tokenize(text):
    return text.split()

text_data = load_text_file("text/dataset.txt")
tokenized = [tokenize(line) for line in text_data]
print("📄 Tokenized text sample:", tokenized[0])

# --- IMAGE ---
def load_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
             if filename.endswith(".jpg") or filename.endswith(".png"):
                  img = Image.open(os.path.join(folder_path, filename)).convert("RGB")
                  images.append(img)
        return images

def preprocess_image(img, size=(224, 224)):
     img_resized = img.resize(size)
     arr = np.array(img_resized, dtype=np.float32) / 225.0
     return arr

image_list = load_images_from_folder("images/")
preprocessed_images = [preprocess_image(img) for img in image_list]
print("🖼️ Image shape:", preprocessed_images[0].shape)

# --- AUDIO ---
def load_audio_file(file_path, sr=16000):
     audio, sample_rate = librosa.load(file_path, sr=sr)
     return audio, sample_rate

def extract_mfcc(audio, sr=16000, n_mfcc=13):
     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
     return mfccs

audio_data, sr = load_audio_file("audio/example.wav")
mfcc_features = extract_mfcc(audio_data, sr)
print("🔈 MFCC shape:", mfcc_features.shape)