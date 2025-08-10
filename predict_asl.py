import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import pyttsx3
import joblib

# Load the trained model and label encoder
model = tf.keras.models.load_model("scripts/asl_letters_cnn.h5")
encoder = joblib.load("scripts/label_encoder_letters.pkl")

engine = pyttsx3.init()

# Initialize sentence builder
sentence = ""
last_prediction = ""
frame_count = 0
pred_stable_count = 10  # number of frames to wait before accepting same prediction

print("🎥 Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # ✅ Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = roi_normalized.reshape(1, 64, 64, 1)

    # Predict
    predictions = model.predict(roi_reshaped)
    pred_index = np.argmax(predictions)
    pred_label = encoder.inverse_transform([pred_index])[0]

    # Wait for stable predictions to avoid flickering
    if pred_label == last_prediction:
        frame_count += 1
    else:
        frame_count = 0
        last_prediction = pred_label

    if frame_count == pred_stable_count:
        if pred_label == "space":
            sentence += " "
        elif pred_label == "del":
            sentence = sentence[:-1]
        elif pred_label == "nothing":
            pass
        else:
            sentence += pred_label

        frame_count = 0  # Reset to avoid repeat add

    # Display outputs
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        engine.say(sentence)
        engine.runAndWait()
    elif key == ord('c'):
        sentence = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
