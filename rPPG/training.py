import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Video capture for collecting training data
cap = cv2.VideoCapture('path_to_training_video.mp4')  # Use a video with known heart rate

# Parameters
some_window_size = 30
features = []
labels = []  # Replace with actual heart rate data corresponding to each window of frames

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
            # Append your corresponding heart rate here
            labels.append(your_heart_rate_value)  # Replace with actual heart rate

            # Reduce the size of features for the next window
            features = features[-some_window_size:]

# Prepare dataset
X = np.array(features).reshape(-1, some_window_size)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Save the model if needed
import joblib
joblib.dump(model, 'rppg_model.pkl')

cap.release()
