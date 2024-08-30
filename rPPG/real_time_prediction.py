import cv2
import numpy as np
import joblib
from feature_extraction import extract_features_from_frame

# Load the trained model and scaler
model = joblib.load('rppg_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters
window_size = 30
features = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features
    frame_features = extract_features_from_frame(frame)
    features.append(frame_features)
    
    if len(features) >= window_size:
        window_features = np.array(features[-window_size:]).flatten().reshape(1, -1)
        window_features = scaler.transform(window_features)
        prediction = model.predict(window_features)
        
        # Display the prediction
        cv2.putText(frame, f'Heart Rate: {prediction[0]:.2f} BPM', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Heart Rate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
