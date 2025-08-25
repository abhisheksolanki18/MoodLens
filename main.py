# main.py

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
from tensorflow.keras.models import load_model
from model.emotion_labels import emotion_labels

# 🎯 Load trained model
model = load_model('model/emotion_model.h5')

# 🎤 Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Set speaking speed

# 😐 To prevent repeating the same emotion constantly
last_prediction = None
speak_delay = 30  # frames to wait before speaking again
frame_counter = 0

# 📸 Initialize webcam
cap = cv2.VideoCapture(0)

# 🧠 Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

print("🔍 Starting real-time emotion detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image horizontally for natural selfie view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb_frame)

    if result.detections:
        for detection in result.detections:
            # Extract bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure square and safe crop
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y + height, x:x + width]

            try:
                # Preprocess face image
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

                # Predict emotion
                prediction = model.predict(reshaped_face, verbose=0)
                label_index = int(np.argmax(prediction))
                emotion = emotion_labels[label_index]

                # Draw bounding box + emotion
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Voice output with frame delay
                if emotion != last_prediction and frame_counter >= speak_delay:
                    engine.say(f"You look {emotion.lower()}")
                    engine.runAndWait()
                    last_prediction = emotion
                    frame_counter = 0

            except Exception as e:
                print("Face preprocessing error:", e)

    cv2.imshow("Emotion Detection AI", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

    frame_counter += 1

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()