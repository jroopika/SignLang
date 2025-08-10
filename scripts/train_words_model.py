"""
train_words_model.py
Memory-friendly training for word-level sign dataset saved as:
  data/words_dataset/processed/X.npy  (np.object array, each element: np.array(frames) shape=(T, H, W, C))
  data/words_dataset/processed/y.npy  (np.array of filenames or labels)
This script uses a generator + tf.data pipeline to pad/truncate sequences on the fly.
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ====== CONFIG ======
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # scripts/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)                # project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "words_dataset", "processed")

TIMESTEPS = 60               # max frames per sample (pad/truncate to this)
H, W, C = 64, 64, 1          # frame size and channels (your preprocessing used these)
BATCH_SIZE = 8               # lower if OOM
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
ENCODER_PATH = os.path.join(MODEL_DIR, "word_label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "words_model_best.h5")

# ====== HELPERS ======
def load_raw_data():
    """Load X (object array) and y (.npy files)."""
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Expected files: {X_path} and {y_path}")
    X = np.load(X_path, allow_pickle=True)  # object array: each X[i] is np.array of frames (T, H, W, C)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def preprocess_frame_sequence(seq, timesteps=TIMESTEPS, h=H, w=W, c=C):
    """
    seq: np.array of shape (T, H, W, C) or (T, H, W) (grayscale maybe)
    returns: np.array shape (timesteps, h, w, c), dtype=float32, normalized [0,1]
    pads with zeros or truncates.
    """
    # ensure channel dimension
    if seq.ndim == 3:  # (T, H, W)
        seq = np.expand_dims(seq, axis=-1)
    # resize frames if shape differs (defensive)
    if seq.shape[1] != h or seq.shape[2] != w:
        # resize each frame (slower) — fallback, but your prepare step should have resized already
        resized = []
        for frame in seq:
            f = tf.image.resize(frame, (h, w)).numpy()
            if f.shape[-1] != c:
                f = f[..., :c] if f.shape[-1] > c else np.concatenate([f, np.zeros((*f.shape[:2], c - f.shape[-1]))], axis=-1)
            resized.append(f)
        seq = np.array(resized, dtype=np.float32)
    # normalize and convert dtype
    seq = seq.astype(np.float32) / 255.0

    t = seq.shape[0]
    if t >= timesteps:
        return seq[:timesteps]   # truncate
    else:
        pad_len = timesteps - t
        pad = np.zeros((pad_len, h, w, c), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

# ====== Load dataset & label encoder ======
print("[INFO] Loading raw data...")
X_raw, y_raw = load_raw_data()
print(f"[INFO] Loaded {len(X_raw)} samples (object array). Example seq shape: {X_raw[0].shape}")
print(f"[INFO] Example label entry: {y_raw[0]}")

# Build label encoder from y_raw (y_raw could be filenames; map to class names if needed)
# If y_raw are filenames (e.g. 'hello_0001.mp4') and you prefer to extract word,
# modify mapping below. For now we treat y_raw entries as labels themselves.
le = LabelEncoder()
y_labels = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print(f"[INFO] Found {num_classes} classes.")
# save encoder
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"[INFO] Saved label encoder -> {ENCODER_PATH}")

# ====== Generator that yields (padded_sequence, label_int) ======
def data_generator(X_obj, y_arr):
    """Generator yields tuple (sequence_np, label_int)."""
    n = len(X_obj)
    for i in range(n):
        seq = X_obj[i]                 # variable-length array
        lbl = y_arr[i]                 # integer label (already encoded)
        # produce padded / truncated sequence
        seq_processed = preprocess_frame_sequence(seq, timesteps=TIMESTEPS, h=H, w=W, c=C)
        yield seq_processed, np.int32(lbl)

# Create tf.data.Dataset
print("[INFO] Creating tf.data pipeline...")
output_signature = (
    tf.TensorSpec(shape=(TIMESTEPS, H, W, C), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
)

ds = tf.data.Dataset.from_generator(
    lambda: data_generator(X_raw, y_labels),
    output_signature=output_signature
)

ds = ds.shuffle(buffer_size=2048) \
       .batch(BATCH_SIZE) \
       .prefetch(AUTOTUNE)

# ====== Build model (Conv3D + LSTM) ======
print("[INFO] Building model...")
# input shape for model: (timesteps, H, W, C)
inp = layers.Input(shape=(TIMESTEPS, H, W, C))

# Use Conv3D to learn spatio-temporal features
x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inp)
x = layers.MaxPooling3D((1, 2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling3D((1, 2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)

# collapse spatial dims per timestep using TimeDistributed(Flatten)
x = layers.TimeDistributed(layers.Flatten())(x)  # shape -> (batch, timesteps, features)

# BiLSTM on top
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ====== Callbacks ======
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
]

# ====== Train ======
print("[INFO] Starting training...")
# We don't have explicit X/y arrays; compute steps_per_epoch
steps_per_epoch = max(1, len(X_raw) // BATCH_SIZE)
validation_split = 0.2
# Simple approach: split indices and create two datasets (train/val) using generator wrappers.

# create indices
n_samples = len(X_raw)
indices = np.arange(n_samples)
np.random.seed(42)
np.random.shuffle(indices)
split = int(n_samples * (1 - validation_split))
train_idx = indices[:split]
val_idx = indices[split:]

def gen_for_indices(idxs):
    for i in idxs:
        seq = X_raw[i]
        lbl = y_labels[i]
        yield preprocess_frame_sequence(seq, timesteps=TIMESTEPS, h=H, w=W, c=C), np.int32(lbl)

train_ds = tf.data.Dataset.from_generator(
    lambda: gen_for_indices(train_idx),
    output_signature=output_signature
).shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: gen_for_indices(val_idx),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("[INFO] Training finished. Best model at:", MODEL_PATH)
