import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import emoji

# Load model and encoder
model = tf.keras.models.load_model('models/model.h5')
encoder = joblib.load('data/processed/label_encoder.pkl')

# Custom emoji mapping (extend as needed per paper's unique approach)
EMOJI_MAP = {
    'A': emoji.emojize(':apple:'), 'B': emoji.emojize(':banana:'),
    'C': emoji.emojize(':cat:'), 'D': emoji.emojize(':dog:'),
    'E': emoji.emojize(':elephant:'), 'F': emoji.emojize(':fish:'),
    'G': emoji.emojize(':grapes:'), 'H': emoji.emojize(':horse:'),
    'I': emoji.emojize(':ice_cream:'), 'J': emoji.emojize(':joy:'),
    'K': emoji.emojize(':key:'), 'L': emoji.emojize(':lemon:'),
    'M': emoji.emojize(':moon:'), 'N': emoji.emojize(':nose:'),
    'O': emoji.emojize(':orange:'), 'P': emoji.emojize(':penguin:'),
    'Q': emoji.emojize(':queen:'), 'R': emoji.emojize(':rainbow:'),
    'S': emoji.emojize(':sun:'), 'T': emoji.emojize(':tree:'),
    'U': emoji.emojize(':umbrella:'), 'V': emoji.emojize(':violet:'),
    'W': emoji.emojize(':watermelon:'), 'X': emoji.emojize(':x:'),
    'Y': emoji.emojize(':yarn:'), 'Z': emoji.emojize(':zebra:'),
    'HELLO': emoji.emojize(':wave:'), 'THANKYOU': emoji.emojize(':pray:'),
    # Add more words from WLASL as needed
}
for label in encoder.classes_:
    if label not in EMOJI_MAP:
        EMOJI_MAP[label] = label  # Default to label if no emoji

st.title("Sign Language Interpreter")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    if st.button("Login"):
        st.session_state.logged_in = True
        st.rerun()
else:
    page = st.sidebar.selectbox("Menu", ["Real-Time", "Video Upload", "Emoji Translation"])

    if page == "Real-Time":
        st.header("Real-Time Sign Detection")
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        sentence = ""
        frame_buffer = []
        last_pred = ""
        frame_count = 0
        pred_stable_count = 10
        while run:
            ret, frame = cap.read()
            if ret:
                roi = frame[100:300, 100:300]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64))
                frame_buffer.append(roi_resized)
                if len(frame_buffer) > 60:
                    frame_buffer.pop(0)
                if len(frame_buffer) == 60:
                    seq = np.array(frame_buffer) / 255.0
                    seq = np.expand_dims(seq, axis=(0, -1))
                    pred = model.predict(seq)
                    pred_label = encoder.inverse_transform([np.argmax(pred)])[0]
                    if pred_label == last_pred:
                        frame_count += 1
                    else:
                        frame_count = 0
                        last_pred = pred_label
                    if frame_count == pred_stable_count and pred_label not in ["space", "del", "nothing"]:
                        sentence += pred_label + " "
                        frame_count = 0
                cv2.putText(frame, f"Pred: {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Sentence: {sentence}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                FRAME_WINDOW.image(frame, channels="BGR")
        cap.release()

    elif page == "Video Upload":
        st.header("Upload Video")
        video_file = st.file_uploader("Upload .mp4 file", type=["mp4"])
        if video_file:
            with open("temp.mp4", "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture("temp.mp4")
            frames = []
            while len(frames) < 60 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                roi = frame[100:300, 100:300]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64))
                frames.append(roi_resized)
            cap.release()
            if len(frames) > 0:
                seq = np.array(frames[:60]) / 255.0
                seq = np.expand_dims(seq, axis=(0, -1))
                pred = model.predict(seq)
                pred_label = encoder.inverse_transform([np.argmax(pred)])[0]
                emoji_text = EMOJI_MAP.get(pred_label, pred_label)
                st.write(f"Detected: {pred_label} (Emoji: {emoji_text})")

    elif page == "Emoji Translation":
        st.header("Emoji Translation")
        text = st.text_input("Enter text (e.g., from detection)")
        if text:
            words = text.split()
            emoji_text = " ".join(EMOJI_MAP.get(word, word) for word in words)
            st.write(f"Emoji: {emoji_text}")