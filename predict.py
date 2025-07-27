import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("models/sign_model.h5")

# Define class labels (manually copy them in the order your dataset has)
class_names = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
])

# Input shape
IMG_SIZE = 64
SEQUENCE_LENGTH = 1  # Since we trained with 1-frame sequences

# Store previous predictions (optional for smoother display)
pred_queue = deque(maxlen=5)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip & crop
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]

    # Preprocess
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)
    img = np.expand_dims(img, axis=1)  # (1, 1, 64, 64, 3)

    # Predict
    preds = model.predict(img)
    pred_label = class_names[np.argmax(preds)]
    pred_queue.append(pred_label)

    # Most frequent label in the last few frames
    final_label = max(set(pred_queue), key=pred_queue.count)

    # Draw results
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.putText(frame, f'Prediction: {final_label}', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show
    cv2.imshow("Sign Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
