# MoodLens – Real-Time Facial Emotion Detection with Voice Feedback

MoodLens is an AI-powered **real-time facial emotion detection** system built using **Python**, **OpenCV**, **MediaPipe**, **TensorFlow/Keras**, and **pyttsx3** for voice feedback.  
It detects human emotions from live camera feed and audibly announces them, making it both **interactive** and **accessible**.

---

## Features

- **Real-Time Detection** – Uses webcam feed to detect faces and classify emotions instantly.
- **Voice Feedback** – Announces detected emotions using the `pyttsx3` text-to-speech engine.
- **Multiple Emotions Supported** – Detects emotions like Happy, Sad, Angry, Surprised, Neutral, etc.
- **Custom Dataset Support** – Easily replace with your own dataset for fine-tuning.
- **Lightweight & Fast** – Optimized for real-time inference using TensorFlow/Keras.
- **Cross-Platform** – Works on Windows, macOS, and Linux.

---

## Technologies Used

- **Python** – Core programming language.
- **OpenCV** – For face detection and real-time video feed handling.
- **MediaPipe** – For accurate and lightweight face landmark detection.
- **TensorFlow & Keras** – For building and training the deep learning model.
- **NumPy** – For numerical operations.
- **pyttsx3** – For converting detected emotions to voice output.
- **Custom CNN Model** – Pre-trained and saved as `emotion_model.h5`.

---

## Requirements

Before running MoodLens, install the required dependencies:

```bash
pip install opencv-python mediapipe tensorflow numpy pyttsx3
```

---

## Implementation Process

1️⃣ Prepare Dataset

- **Download an emotion detection dataset** (https://www.kaggle.com/datasets/msambare/fer2013).

- **Organize images into**:

-**dataset/train/<emotion_name>/**

-**dataset/test/<emotion_name>/**


2️⃣ Train the Model

- **Run the training script**:
```bash
python train_model.py
```

**This will**:

-**Load and preprocess images.**

-**Train a CNN model on the dataset.**

-**Save the trained model as model/emotion_model.h5.**



3️⃣ Run Real-Time Detection

**Start the main script**:
```bash
python main.py
```
**The program will**:

-**Open webcam feed.**

-**Detect faces using MediaPipe.**

-**Predict emotion using the trained model.**

-**Announce detected emotion via voice.**

---

## How It Works

- **Face Detection – MediaPipe detects and crops the face region from the webcam feed.**
- **Preprocessing – The cropped face is resized and normalized for model input.**
- **Emotion Classification – The trained CNN predicts the probability for each emotion.**
- **Voice Output – The top predicted emotion is spoken aloud using pyttsx3.**
- **Real-Time Display – The detected emotion is displayed on the webcam window.**

---
## Project Folder Structure

```
MoodLens/
├── dataset/
│   ├── train/
│   └── test/
├── model/
│   ├── emotion_labels.py
│   └── emotion_model.h5
├── train_model.py
├── main.py
```
---
## Demo 👉🏻 [photo](file_00000000238461f89d57ab2f469fbf53.png)
---
## Team Members
- [Aditya Kumar](https://github.com/baisoyaaditya)
- [Abhishek Solanki](https://github.com/abhisheksolanki18)
- [Ankit Kushwaha](https://github.com/ankitthakur7)
- [Abhishek Parmar](https://github.com/abhishekparmar2005)
