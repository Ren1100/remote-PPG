import cv2
import numpy as np
import dlib
import time
import csv
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
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

left_expand_ratio = 0.25
top_expand_ratio = 0.25

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Output CSV File
csv_filename = "rPPG_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "rPPG_filtered", "BPM"])  # CSV Header

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

# Create a named window
cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)

# Move window to the center
cv2.moveWindow('Webcam Feed', 500, 100)

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
        
        # Get the number of face pixels in the skin-segmented region
        n_facepixels = np.sum(masked_face // 255)
        # # Display the segmented face result
        # frame_toshow = cv2.cvtColor(skin_segmented_face, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame_toshow)
        # plt.show()
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
    
    l = int(fps * 1.6)
    if mean_rgb.shape[0] > l:
        # Apply POS algorithm
        rPPG_signals = POS(mean_rgb, l)
        # Apply bandpass filter
        rPPG_filtered = bandpass_filter(rPPG_signals, fps)
        # Standardization
        rPPG_filtered = standardization_signal(rPPG_filtered)
        csv_writer.writerow([current_time, rPPG_filtered[-1], bpm_display])
    current_time = time.time()
    if current_time >= bpm_estimation_start:
        if (current_time - start_time) >= 2.0:
            start_time = current_time  # Reset timer
            # Process the mean RGB signal for rPPG
                # estimate BPM using frequncy analysis 
            rPPG_bpm = BPM_estimation(rPPG_filtered, n_segment, fps)
            print(f"BPM: {rPPG_bpm:.2f}")
            bpm_display = f"BPM: {rPPG_bpm:.2f}"  # Update the BPM display text
    cv2.putText(resized_frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(frame)

    cv2.imshow('Webcam Feed', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  # Press 'q' to exit the loop

cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()