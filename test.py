import cv2
import numpy as np
import dlib
import time
from scipy import signal
from scipy.signal import welch
from signal_processing import *
from BPM_estimation import *
from POS import *
from skin_segment import *

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

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

delay = 10
bpm_display = "BPM: N/A"  # Initialize BPM display text

# Define time duration for recording (in seconds)
start_time = time.time()
bpm_estimation_start = start_time + delay  # Set the initial BPM estimation time

# Frame count
mean_rgb = np.empty((0, 3))  # Initialize mean_rgb array

while True:
    ret, frame = cap.read()
    
    # Check if the frame was captured correctly
    if not ret or frame is None:
        print("Failed to capture video.")
        break

    h, w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if f_cnt == 0:
        rects = detector(gray_frame, 0)
        if len(rects) > 0:
            rect = rects[0]  # Get the first detected face
            left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

            width = abs(right - left)
            height = abs(bottom - top)
            face_left = int(left - (left_expand_ratio / 2 * width))
            face_top = int(top - (top_expand_ratio / 2 * height))
            face_right = right
            face_bottom = bottom
        else:
            print("No face detected.")
            continue  # Skip to the next iteration if no face is detected

    cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)

    face = frame[face_top:face_bottom, face_left:face_right]

    # # Convert the face region to YCrCb color space
    # face_YCrCb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    # # Define the skin color range in YCrCb
    # lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    # upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # # Create a binary mask where skin color is white and the rest is black
    # mask = cv2.inRange(face_YCrCb, lower_skin, upper_skin)
    # n_skinpixels = np.sum(mask)

    # if n_skinpixels == 0:
    #     print("No skin pixels detected.")
    #     continue  # Skip this iteration if no skin pixels are detected

    # # Apply the mask to the face region
    # masked_face = cv2.bitwise_and(face, face, mask=mask)

    masked_face, n_skinpixels = YCrCB_Segment(face)

    if n_skinpixels == 0 or masked_face is None:
        print("No skin pixels detected.")
        continue  # Skip this iteration if no skin pixels are detected

    # Get the mean RGB value in the skin
    mean_r = np.sum(masked_face[:, :, 2]) / n_skinpixels
    mean_g = np.sum(masked_face[:, :, 1]) / n_skinpixels
    mean_b = np.sum(masked_face[:, :, 0]) / n_skinpixels

    mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b]))) if f_cnt > 0 else np.array([[mean_r, mean_g, mean_b]])

    f_cnt += 1

    current_time = time.time()
    if current_time >= bpm_estimation_start:
        if (current_time - start_time) >= 2.0:
            start_time = current_time  # Reset timer

            # Process the mean RGB signal for rPPG
            l = int(fps * 1.6)
            if mean_rgb.shape[0] > l:
                # Apply POS algorithm
                rPPG_signals = POS(mean_rgb, l)

                # Apply bandpass filter
                rPPG_filtered = bandpass_filter(rPPG_signals, fps)

                # Standardization
                rPPG_filtered = standardization_signal(rPPG_filtered)

                # estimate BPM using frequncy analysis 
                rPPG_bpm = BPM_estimation(rPPG_filtered, n_segment, fps)
                print(f"BPM: {rPPG_bpm:.2f}")
                bpm_display = f"BPM: {rPPG_bpm:.2f}"  # Update the BPM display text

    cv2.putText(frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
