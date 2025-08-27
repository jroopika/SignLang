import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import time
from sklearn.utils import shuffle
import gc  # For explicit garbage collection

# Paths
LETTER_DIR = "data/letters_dataset/asl_alphabet_train"
VIDEO_DIR = "data/words_dataset/videos"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)
IMG_SIZE = (64, 64)
SEQ_LENGTH = 60  # Fixed sequence length
BATCH_SIZE = 100  # Reduced batch size to avoid memory issues

# Data holders
X_batch = []
y_batch = []
batch_idx = 0

start_time = time.time()

# Process letters (repeat frame to simulate sequence)
for label in sorted(os.listdir(LETTER_DIR)):
    class_dir = os.path.join(LETTER_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    print(f"📂 Processing letter class '{label}' ...")
    batch_start_time = time.time()
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue
        img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        seq = np.repeat(img[np.newaxis, ...], SEQ_LENGTH, axis=0)
        X_batch.append(seq.astype(np.float32))
        y_batch.append(label)
        if len(X_batch) >= BATCH_SIZE:
            np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), np.array(X_batch, dtype=np.float32))
            np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), np.array(y_batch))
            print(f"[TIMER] Batch {batch_idx} processed in {time.time() - batch_start_time:.2f} seconds.")
            batch_start_time = time.time()
            X_batch = []
            y_batch = []
            batch_idx += 1
            # Force garbage collection
            gc.collect()
    # Save any remaining samples in the last batch
    if X_batch:
        np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), np.array(X_batch, dtype=np.float32))
        np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), np.array(y_batch))
        print(f"[TIMER] Batch {batch_idx} processed in {time.time() - batch_start_time:.2f} seconds.")
        X_batch = []
        y_batch = []
        batch_idx += 1
        gc.collect()

# Process words (use frame segmentation from WLASL_v0.3.json)
with open("data/words_dataset/WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

X_batch = []
y_batch = []
batch_start_time = time.time()
for entry in wlasl_data:
    gloss = entry["gloss"]
    for inst in entry["instances"]:
        video_id = inst["video_id"]
        video_file = f"{video_id}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_file)
        if os.path.exists(video_path):
            print(f"[INFO] Processing {video_file}...")
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_start = inst.get("frame_start", 1) - 1
            frame_end = inst.get("frame_end", frame_count)
            if frame_end is None or frame_end <= 0:
                frame_end = frame_count
            print(f"[DEBUG] {video_file}: frame_start={frame_start}, frame_end={frame_end}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            frames = []
            max_failed_reads = 10
            failed_reads = 0
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Could not read first frame of {video_file}")
                cap.release()
                continue
            else:
                print(f"[DEBUG] First frame of {video_file} read successfully.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            for frame_idx in range(int(frame_end - frame_start)):
                ret, frame = cap.read()
                if not ret:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        print(f"[ERROR] Skipping {video_file}: too many failed frame reads.")
                        break
                    continue
                failed_reads = 0
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, IMG_SIZE).astype(np.float32) / 255.0
                    frame = np.expand_dims(frame, axis=-1)
                    frames.append(frame)
                except Exception as e:
                    print(f"[ERROR] Skipping frame in {video_file}: {e}")
                    continue
                if len(frames) >= SEQ_LENGTH:
                    break
            cap.release()
            if len(frames) == 0:
                print(f"[ERROR] No valid frames for {video_file}, skipping.")
                continue
            while len(frames) < SEQ_LENGTH:
                frames.append(frames[-1])
            X_batch.append(np.array(frames[:SEQ_LENGTH], dtype=np.float32))
            y_batch.append(gloss)
            if len(X_batch) >= BATCH_SIZE:
                np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), np.array(X_batch, dtype=np.float32))
                np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), np.array(y_batch))
                print(f"[TIMER] Batch {batch_idx} processed in {time.time() - batch_start_time:.2f} seconds.")
                batch_start_time = time.time()
                X_batch = []
                y_batch = []
                batch_idx += 1
                gc.collect()
    # Save any remaining samples in the last batch
    if X_batch and len(X_batch) > 1:  # At least 2 samples for train_test_split
        np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_idx}.npy"), np.array(X_batch, dtype=np.float32))
        np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_idx}.npy"), np.array(y_batch))
        print(f"[TIMER] Batch {batch_idx} processed in {time.time() - batch_start_time:.2f} seconds.")
        X_batch = []
        y_batch = []
        batch_idx += 1
        gc.collect()

# Scan all batches for unique labels
all_labels = set()
for i in range(batch_idx):
    try:
        y_batch = np.load(os.path.join(PROCESSED_DIR, f"y_batch_{i}.npy"))
        all_labels.update(y_batch.tolist())
        del y_batch
        gc.collect()
    except:
        print(f"Warning: Could not load y_batch_{i}.npy")

# Fit LabelEncoder on all labels
le = LabelEncoder()
le.fit(list(all_labels))
joblib.dump(le, os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
print(f"Label encoder saved with {len(le.classes_)} classes")

# Memory-efficient batching for train/test split
train_idx = 0
test_idx = 0

for i in range(batch_idx):
    try:
        X_batch = np.load(os.path.join(PROCESSED_DIR, f"X_batch_{i}.npy"))
        y_batch = np.load(os.path.join(PROCESSED_DIR, f"y_batch_{i}.npy"))
        y_batch_encoded = le.transform(y_batch)
        
        # Shuffle batch
        X_batch, y_batch_encoded = shuffle(X_batch, y_batch_encoded, random_state=42)
        
        # Split batch
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_batch, y_batch_encoded, test_size=0.2, random_state=42, stratify=y_batch_encoded)
        
        # Save split batches directly to disk
        np.save(os.path.join(PROCESSED_DIR, f"X_train_batch_{train_idx}.npy"), X_train_b)
        np.save(os.path.join(PROCESSED_DIR, f"y_train_batch_{train_idx}.npy"), y_train_b)
        np.save(os.path.join(PROCESSED_DIR, f"X_test_batch_{test_idx}.npy"), X_test_b)
        np.save(os.path.join(PROCESSED_DIR, f"y_test_batch_{test_idx}.npy"), y_test_b)
        
        # Free memory and clean up
        del X_batch, y_batch, y_batch_encoded, X_train_b, X_test_b, y_train_b, y_test_b
        gc.collect()
        
        train_idx += 1
        test_idx += 1
        os.remove(os.path.join(PROCESSED_DIR, f"X_batch_{i}.npy"))
        os.remove(os.path.join(PROCESSED_DIR, f"y_batch_{i}.npy"))
        print(f"Processed batch {i}/{batch_idx}")
    except Exception as e:
        print(f"Error processing batch {i}: {e}")

# Memory-mapped array concatenation function
def memory_efficient_concat_and_save(prefix, out_name):
    """Memory-efficient concatenation using memmap"""
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith(prefix)])
    if not files:
        print(f"No files found with prefix {prefix}")
        return
    
    print(f"Concatenating {len(files)} files with prefix {prefix}...")
    
    # First determine total shape
    total_samples = 0
    sample_shape = None
    sample_dtype = None
    
    for f in files:
        try:
            arr = np.load(os.path.join(PROCESSED_DIR, f), mmap_mode='r')
            if sample_shape is None:
                if len(arr.shape) > 1:
                    sample_shape = arr.shape[1:]
                else:
                    sample_shape = ()
                sample_dtype = arr.dtype
            total_samples += len(arr)
            del arr
        except Exception as e:
            print(f"Error checking file {f}: {e}")
    
    if total_samples == 0 or sample_shape is None:
        print(f"No valid data found with prefix {prefix}")
        return
    
    # Create memory-mapped output file
    output_path = os.path.join(PROCESSED_DIR, out_name)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Create the shape based on whether it's X data or y data
    final_shape = (total_samples,) + sample_shape
    
    try:
        # Create memory map file
        mmap_file = np.lib.format.open_memmap(
            output_path, mode='w+', dtype=sample_dtype, shape=final_shape)
        
        # Copy data in chunks
        idx = 0
        for i, f in enumerate(files):
            try:
                print(f"Processing file {i+1}/{len(files)}: {f}")
                arr = np.load(os.path.join(PROCESSED_DIR, f))
                mmap_file[idx:idx+len(arr)] = arr
                idx += len(arr)
                os.remove(os.path.join(PROCESSED_DIR, f))
                del arr
                gc.collect()
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        
        # Flush to ensure all data is written
        mmap_file.flush()
        del mmap_file
        gc.collect()
        print(f"Successfully saved {output_path} with {total_samples} samples")
    except Exception as e:
        print(f"Error creating output file: {e}")

print("Starting final concatenation process...")
memory_efficient_concat_and_save("X_train_batch_", "X_train.npy")
memory_efficient_concat_and_save("y_train_batch_", "y_train.npy")
memory_efficient_concat_and_save("X_test_batch_", "X_test.npy")
memory_efficient_concat_and_save("y_test_batch_", "y_test.npy")

print(f"[TIMER] Total preprocessing time: {time.time() - start_time:.2f} seconds.")
print("✅ Preprocessing done!")
print(f"   - Classes: {len(le.classes_)} -> {list(le.classes_)}")