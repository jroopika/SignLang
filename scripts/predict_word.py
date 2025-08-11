import pickle
import numpy as np
import tensorflow as tf
import cv2
import time

# Load class list for mapping
class_list_path = r"C:\Users\JULURU ROOPIKA\OneDrive\Documents\Sign_language_Interpreter\data\words_dataset\wlasl_class_list.txt"
class_idx_to_word = {}
with open(class_list_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            idx, word = line.strip().split("\t", 1)
            class_idx_to_word[int(idx)] = word

print("[INFO] Loading model...")
model = tf.keras.models.load_model(
    r"C:\Users\JULURU ROOPIKA\OneDrive\Documents\Sign_language_Interpreter\models\words_model_best.h5"
)
print(f"[INFO] Model loaded from words_model_best.h5")

with open(
    r"C:\Users\JULURU ROOPIKA\OneDrive\Documents\Sign_language_Interpreter\models\word_label_encoder.pkl", "rb"
) as f:
    label_encoder = pickle.load(f)
print(f"[INFO] Label encoder loaded.")

# -----------------------------
# Webcam Real-time Prediction
# -----------------------------
FRAME_SIZE = (64, 64)
MAX_FRAMES = 60
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open webcam.")

frames = []
pred_label = ""

print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FRAME_SIZE)
    frames.append(resized)

    # Draw prediction on frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Prediction: {pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sign Language Word Prediction", display_frame)

    # Predict every MAX_FRAMES
    if len(frames) == MAX_FRAMES:
        input_frames = np.array(frames) / 255.0
        input_frames = np.expand_dims(input_frames, axis=-1)  # (60, 64, 64, 1)
        input_frames = np.expand_dims(input_frames, axis=0)   # (1, 60, 64, 64, 1)
        pred = model.predict(input_frames)
        pred_class_idx = np.argmax(pred)
        # Try to map using label encoder classes (if numeric or filename)
        try:
            label = label_encoder.inverse_transform([pred_class_idx])[0]
            # If label is numeric, map to word
            if label.isdigit():
                word = class_idx_to_word.get(int(label), label)
            else:
                # If label is a filename, try to extract numeric part
                num_part = label.split(".")[0]
                if num_part.isdigit():
                    word = class_idx_to_word.get(int(num_part), label)
                else:
                    word = label
        except Exception:
            word = class_idx_to_word.get(pred_class_idx, str(pred_class_idx))
        pred_label = word
        frames = []  # reset buffer

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
