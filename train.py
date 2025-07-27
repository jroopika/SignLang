import sys
import os
sys.path.append(os.path.abspath('.'))

from utils.data_loader import load_dataset
from models.cnn_lstm_model import build_cnn_lstm_model
import tensorflow as tf

data_dir = "dataset/asl_alphabet_train"
batch_size = 32
img_size = (64, 64)

# Load data
train_ds, class_names = load_dataset(data_dir, img_size, batch_size)

# Expand dims to add time_steps=1 for LSTM input
train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, axis=1), y))

# Build model
model = build_cnn_lstm_model((1, img_size[0], img_size[1], 3), len(class_names))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_ds, epochs=10)
model.save("models/sign_model.h5")
