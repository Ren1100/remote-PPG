import cv2
import numpy as np
import joblib
from feature_extraction import extract_features_from_frames

# Load the trained model
model = joblib.load('rppg_model.pkl')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters
window_size = 30
features = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Extract features
    frame_features = extract_features_from_frames(frame_dir)  # Adjust frame_dir or method as needed
    avg_features = np.mean(frame_features, axis=0)  # Example aggregation

    features.append(avg_features)
    if len(features) >= window_size:
        window_features = np.array(features[-window_size:]).flatten().reshape(1, -1)
        prediction = model.predict(window_features)
        cv2.putText(frame, f'Heart Rate: {prediction[0]:.2f} BPM', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Heart Rate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
