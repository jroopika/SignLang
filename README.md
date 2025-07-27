# 🤟 Sign Language Recognition and Translation (ASL) 🧠📷

This project is a deep learning-based system that recognizes American Sign Language (ASL) alphabets using a **CNN + LSTM** architecture built with **TensorFlow**.

It takes in image data of hand signs and classifies them into their corresponding alphabet classes. This version uses the **ASL Alphabet Dataset** for training and prediction.

---
### Dataset
This project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).  
Download it manually and extract it inside the `dataset/` folder.

### Directory Structure
SignLangTranslator/
├── dataset/
│ ├── asl_alphabet_train/ # Training images
│ └── asl_alphabet_test/ # (Optional) test images
├── models/
│ ├── cnn_lstm_model.py # CNN + LSTM model definition
│ └── sign_model.h5 # Trained model weights
├── scripts/
│ └── utils/
│ └── data_loader.py # Dataset loading function
├── train.py # Training script
├── predict.py # Image prediction script
├── test.py # (Optional) for future webcam/test predictions
├── main.py # (Optional) entry point
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Files to ignore in Git
---

## 🚀 How to Run

### 🔧 Step 1: Set up environment
```bash
python -m venv env_311
env_311\Scripts\activate  # or source env_311/bin/activate (Linux/Mac)
pip install -r requirements.txt
🏋️ Step 2: Train the Model
bash
Copy
Edit
python train.py
Uses images from dataset/asl_alphabet_train/

Trains and saves model as models/sign_model.h5

🔎 Step 3: Predict
bash
Copy
Edit
python predict.py
Modify image path inside predict.py or take input from user

Outputs predicted class (A–Z, etc.)

🧠 Model Architecture
CNN layers extract spatial features

LSTM layer learns temporal dependencies (mimicking sequence understanding)

Dense layers classify into 29 ASL classes

📦 Dataset
Dataset: ASL Alphabet Dataset

87000+ images across 29 classes (A–Z + space, delete, nothing)

💻 Tools & Libraries
Python 3.11

TensorFlow 2.x

NumPy

OpenCV (for future real-time webcam support)

