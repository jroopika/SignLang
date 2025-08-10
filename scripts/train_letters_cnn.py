import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# Load data
X_train = np.load("X_train_letters.npy")
X_test = np.load("X_test_letters.npy")
y_train = np.load("y_train_letters.npy")
y_test = np.load("y_test_letters.npy")

# Load the label encoder (for printing class names later)
encoder = joblib.load("label_encoder_letters.pkl")
num_classes = len(encoder.classes_)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# Save model
model.save("asl_letters_cnn.h5")
print("✅ Model trained and saved as 'asl_letters_cnn.h5'")

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 Test accuracy: {test_acc:.4f}")

# Predict a sample
sample_idx = np.random.randint(len(X_test))
sample_img = X_test[sample_idx:sample_idx+1]
pred_class = np.argmax(model.predict(sample_img), axis=1)
pred_label = encoder.inverse_transform(pred_class)[0]

print(f"🔎 Sample prediction: {pred_label}")
