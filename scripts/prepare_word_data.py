import os
import cv2
import numpy as np

# 📁 Paths
video_dir = "../data/words_dataset/videos"
output_dir = "../data/words_dataset/processed"
os.makedirs(output_dir, exist_ok=True)

# 🧠 Data holders
X = []
y = []

# 📦 Label extraction helper (e.g., "hello_0001.mp4" → "hello")
def extract_label(filename):
    return filename.split("_")[0]

# 🔄 Processing each video
for video_file in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video_file)

    if not video_file.endswith(".mp4"):
        continue

    print(f"[INFO] Processing {video_file}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_file}")
        continue

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to 64x64
        frame = cv2.resize(frame, (64, 64))

        # Add extra dimension for channel → (64, 64, 1)
        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"[WARN] No valid frames in: {video_file}")
        continue

    X.append(np.array(frames))  # shape: (timesteps, 64, 64, 1)
    y.append(extract_label(video_file))

    print(f"[INFO] Finished processing {video_file} with {len(frames)} valid frames.")
# 🧠 Save processed data safely
X_array = np.array(X, dtype=object)
np.save(os.path.join(output_dir, "X.npy"), X_array)
np.save(os.path.join(output_dir, "y.npy"), y)

print(f"[✅] Saved {len(X)} video sequences to {output_dir}")
