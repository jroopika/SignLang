import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 🔥 FIX: Point to the folder containing A/, B/, ... directories directly:
DATA_DIR = "../data/letters_dataset/asl_alphabet_train"
IMG_SIZE = 64

images = []
labels = []

for label in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    print(f"📂 Processing class '{label}' ...")
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

if len(images) == 0:
    raise ValueError("❌ No images were loaded! Check your DATA_DIR path or dataset structure.")

# Convert to numpy arrays
X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the encoder for later prediction
joblib.dump(encoder, "label_encoder_letters.pkl")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Save datasets
np.save("X_train_letters.npy", X_train)
np.save("X_test_letters.npy", X_test)
np.save("y_train_letters.npy", y_train)
np.save("y_test_letters.npy", y_test)

print("✅ Letter dataset prepared and saved successfully!")
print(f"   - X_train: {X_train.shape}")
print(f"   - X_test:  {X_test.shape}")
print(f"   - Classes: {len(encoder.classes_)} -> {list(encoder.classes_)}")
    