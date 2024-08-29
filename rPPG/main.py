import cv2
import dlib
import numpy as np
import joblib

# Load the trained model
model = joblib.load('rppg_model.pkl')

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Video capture and feature extraction process
cap = cv2.VideoCapture(0)

# Parameters
some_window_size = 30
features = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        roi = frame[y:y + int(h * 0.3), x:x + w]
        green_channel = roi[:, :, 1]
        avg_green_intensity = green_channel.mean()
        features.append(avg_green_intensity)

        if len(features) >= some_window_size:
            window_features = np.array(features[-some_window_size:]).reshape(1, -1)
            prediction = model.predict(window_features)
            cv2.putText(frame, f'Heart Rate: {prediction[0]:.2f} BPM', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Heart Rate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
