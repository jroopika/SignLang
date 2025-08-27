import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

# Enable mixed precision for faster training if available
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    print("Mixed precision not available")

# Paths
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load labels first (they're small)
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
num_classes = len(np.unique(y_train))

print(f"Dataset has {len(y_train)} training samples and {len(y_test)} test samples with {num_classes} classes")

# Get shapes without loading data
# Using memory-mapped arrays to check shapes without loading full data
X_train_mmap = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"), mmap_mode='r')
X_test_mmap = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"), mmap_mode='r')
print(f"X_train shape: {X_train_mmap.shape}, X_test shape: {X_test_mmap.shape}")

# Create TensorFlow datasets
BATCH_SIZE = 8

# Create training dataset from memory-mapped array
train_dataset = tf.data.Dataset.from_generator(
    lambda: ((X_train_mmap[i], y_train[i]) for i in range(len(y_train))),
    output_signature=(
        tf.TensorSpec(shape=(60, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)  # Added .repeat()

# Create test dataset from memory-mapped array
test_dataset = tf.data.Dataset.from_generator(
    lambda: ((X_test_mmap[i], y_test[i]) for i in range(len(y_test))),
    output_signature=(
        tf.TensorSpec(shape=(60, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)  # Added .repeat()

# Calculate steps per epoch
steps_per_epoch = len(y_train) // BATCH_SIZE
validation_steps = len(y_test) // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# Model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(60, 64, 64, 1)),
    layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),  # Convert grayscale to RGB for MobileNet
    layers.TimeDistributed(base_model),
    layers.TimeDistributed(layers.GlobalAveragePooling2D()),
    layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Use a lower learning rate for better stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'model_best.h5'), monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=25,
    steps_per_epoch=steps_per_epoch,  # Added steps_per_epoch
    validation_steps=validation_steps,  # Added validation_steps
    callbacks=callbacks
)

# Save model
model.save(os.path.join(MODEL_DIR, 'model.h5'))
print("✅ Model trained and saved as 'models/model.h5'")

# Save training history
import pickle
with open(os.path.join(MODEL_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Training history saved as 'models/training_history.pkl'")