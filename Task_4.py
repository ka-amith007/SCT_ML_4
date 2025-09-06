# 📌 Hand Gesture Recognition Model (MediaPipe + RandomForest)

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# --- Step 1: Initialize Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Example dataset (X = landmarks, y = gesture labels)
X, y = [], []

# --- Step 2: Capture Data (Press keys to save gestures) ---
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

print("Press 's' to save sample with label, 'q' to quit")

label = "OpenHand"  # 👈 Change label (e.g., 'Fist', 'Peace', etc.)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save sample when 's' pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                X.append(landmarks)
                y.append(label)
                print(f"Sample saved for {label}")

    cv2.imshow("Hand Gesture Capture", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Step 3: Train Model ---
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    with open("gesture_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved as gesture_model.pkl")
else:
    print("⚠️ No data collected. Please run again and save samples.")
