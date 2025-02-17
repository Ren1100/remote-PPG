import cv2
import numpy as np
import time
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from POS import POS
from BPM_estimation import BPM_estimation
from signal_processing import bandpass_filter, standardization_signal

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Streamlit UI
st.title("Real-time rPPG Estimation with Streamlit")
frame_placeholder = st.empty()
plot_placeholder = st.empty()
bpm_placeholder = st.empty()

# Video Capture
cap = cv2.VideoCapture(0)
fps = 30
f_cnt = 0
mean_rgb = np.empty((0, 3))
start_time = time.time()
bpm_display = "BPM: N/A"

# Streamlit Button to Stop
stop_signal = st.button("Stop Capture")

while cap.isOpened() and not stop_signal:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for faster processing
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height))
    
    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        mask = np.zeros((height, width), dtype=np.uint8)
        landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks.landmark]
        
        # Define ROI using outer face boundary landmarks
        outer_face_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]
        face_boundary_points = np.array([landmark_points[i] for i in outer_face_indices], np.int32)
        cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillConvexPoly(mask, face_boundary_points, 255)
        masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)
        
        # Compute mean RGB
        n_facepixels = np.sum(masked_face // 255)
        if n_facepixels > 0:
            mean_r = np.sum(masked_face[:, :, 2]) / n_facepixels
            mean_g = np.sum(masked_face[:, :, 1]) / n_facepixels
            mean_b = np.sum(masked_face[:, :, 0]) / n_facepixels
            mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))
    
    # Process rPPG and estimate BPM
    l = int(fps * 1.6)
    if mean_rgb.shape[0] > l:
        rPPG_signals = POS(mean_rgb, l)
        rPPG_filtered = bandpass_filter(rPPG_signals, fps)
        rPPG_filtered = standardization_signal(rPPG_filtered)
        bpm = BPM_estimation(rPPG_filtered, 2, fps)
        bpm_display = f"BPM: {bpm:.2f}"
    
    # Update GUI
    frame_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    bpm_placeholder.write(bpm_display)
    
    # Plot rPPG signal
    fig, ax = plt.subplots()
    ax.plot(rPPG_filtered, color='red', label='rPPG Signal')
    ax.set_title("rPPG Signal")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plot_placeholder.pyplot(fig)
    
    # Exit on stop signal
    if stop_signal:
        break

cap.release()
st.write("Capture Stopped.")
