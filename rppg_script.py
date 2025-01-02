import cv2
import numpy as np
import dlib
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from signal_processing import *
from BPM_estimation import *
from POS import *
import mediapipe as mp

# Initialize the face detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
# cv2.imshow('Webcam Feed', frame)
fps = 30
n_segment = 2
left_expand_ratio = 0.25
top_expand_ratio = 0.25

f_cnt = 0
face_left, face_top, face_right, face_bottom = 0, 0, 0, 0
mask = None
n_skinpixels = 0

delay = 15
bpm_display = "BPM: N/A"  # Initialize BPM display text

# Define time duration for recording (in seconds)
start_time = time.time()
bpm_estimation_start = start_time + delay  # Set the initial BPM estimation time

# Frame count
mean_rgb = np.empty((0, 3))  # Initialize mean_rgb array

# Initialize a Matplotlib figure for rPPG signal
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Real-Time rPPG Signal")
ax.set_xlabel("Frames")
ax.set_ylabel("Amplitude")
rPPG_plot, = ax.plot([], [], color='b', label='rPPG Signal')
ax.legend()
rPPG_data = []  # Buffer to store the rPPG signal for plotting
# Update the signal window size (number of samples to display)
window_size = int(fps * 10)  # Show the last 10 seconds of data

while True:
    ret, frame = cap.read()
    
    # Check if the frame was captured correctly
    if not ret or frame is None:
        print("Failed to capture video.")
        break

    scale_percent = 60  # Resize image by 60% for faster processing
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height))

    # Convert the resized image to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Apply face detection and get landmarks
    results = face_mesh.process(rgb_frame)

    # Proceed if face landmarks are detected
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        # Create a mask for the face region using landmarks
        mask = np.zeros((height, width), dtype=np.uint8)

        # Get the coordinates of the landmarks corresponding to the face region
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_points.append((x, y))

        # Use only the landmarks around the outer face boundary for the convex hull
        outer_face_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
            365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        face_boundary_points = np.array([landmark_points[i] for i in outer_face_indices], np.int32)

        cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Create a convex hull around the face boundary points
        cv2.fillConvexPoly(mask, face_boundary_points, 255)

        # Apply more dilation to expand the mask outward
        kernel = np.ones((1, 1), np.uint8)  # Increase kernel size for more dilation
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Apply the mask to extract the full face region
        masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

        cv2.imshow('Masked Face', masked_face)
        cv2.moveWindow('Masked Face', 1200, 500)  # Position the masked face window at (500, 100)


        # Get the number of face pixels in the skin-segmented region
        n_facepixels = np.sum(mask // 255)

        # If face pixels are found, calculate the mean RGB values in the skin-segmented face region
        if n_facepixels > 0:
            mean_r = np.sum(masked_face[:, :, 2]) / n_facepixels
            mean_g = np.sum(masked_face[:, :, 1]) / n_facepixels
            mean_b = np.sum(masked_face[:, :, 0]) / n_facepixels

            if f_cnt == 0:
                mean_rgb = np.array([mean_r, mean_g, mean_b])
                print(mean_rgb)
            else:
                mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))

    # Increment frame counters
    # break
    f_cnt += 1

    # Process the mean RGB signal for rPPG
    l = int(fps * 1.6)
    if mean_rgb.shape[0] > l:
        # Apply POS algorithm
        rPPG_signals = POS(mean_rgb, l)

        # Apply bandpass filter
        rPPG_filtered = bandpass_filter(rPPG_signals, fps)

        # Standardization
        rPPG_filtered = standardization_signal(rPPG_filtered)
        rPPG_data.extend(rPPG_filtered.tolist())
        rPPG_data = rPPG_data[-window_size:]  # Keep only the last 'window_size' samples
        rPPG_plot.set_data(range(len(rPPG_data)), rPPG_data)
        ax.set_xlim(0, len(rPPG_data))
        ax.set_ylim(min(rPPG_data) - 0.1, max(rPPG_data) + 0.1)
        plt.pause(0.001)  # Update the plot

    current_time = time.time()
    if current_time >= bpm_estimation_start:
        if (current_time - start_time) >= 2.0:
            start_time = current_time  # Reset timer

                # estimate BPM using frequncy analysis 
            rPPG_bpm = BPM_estimation(rPPG_filtered, n_segment, fps)
            print(f"BPM: {rPPG_bpm:.2f}")
            bpm_display = f"BPM: {rPPG_bpm:.2f}"  # Update the BPM display text

    cv2.putText(resized_frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Webcam Feed', resized_frame)
    cv2.moveWindow('Webcam Feed', 1200, 100)  # Position the webcam window at (100, 100)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
