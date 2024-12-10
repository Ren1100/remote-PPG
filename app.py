import cv2
import numpy as np
import mediapipe as mp
import time
from signal_processing import bandpass_filter, standardization_signal
from BPM_estimation import BPM_estimation
from POS import POS
import streamlit as st

# Streamlit configuration
st.title("Real-Time BPM Detection with rPPG")
st.write("This app uses a webcam to detect your heart rate in real-time.")

# Streamlit sidebar controls
run_app = st.checkbox("Start Webcam")
fps = 30
n_segment = 2
delay = 15
bpm_display = "BPM: N/A"  # Initialize BPM display text

# Initialize face detection lazily (on checkbox click)
mp_face_mesh = mp.solutions.face_mesh


def initialize_face_mesh():
    """Lazy initialization for face detection."""
    return mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Initialize webcam lazily
cap = None
face_mesh = None
mean_rgb = np.empty((0, 3))


def process_webcam():
    global cap, face_mesh, mean_rgb

    with st.spinner("Initializing webcam and face detection..."):
        face_mesh = initialize_face_mesh()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam. Please check permissions or try again.")
        return

    st.success("Webcam started. Processing video feed...")
    return cap


# Run the app if checkbox is checked
if run_app:
    cap = process_webcam()
    if cap:
        # Main processing loop
        start_time = time.time()
        bpm_estimation_start = start_time + delay

        frame_placeholder = st.empty()  # Placeholder for video stream
        bpm_placeholder = st.empty()  # Placeholder for BPM display

        try:
            while True:
                ret, frame = cap.read()

                # Check if the frame was captured correctly
                if not ret or frame is None:
                    st.write("Failed to capture video.")
                    break

                # Resize image for faster processing
                scale_percent = 60
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                resized_frame = cv2.resize(frame, (width, height))

                # Convert the resized image to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Apply face detection and get landmarks
                results = face_mesh.process(rgb_frame)

                # If face landmarks are detected
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]

                    # Create mask for the face region using landmarks
                    mask = np.zeros((height, width), dtype=np.uint8)
                    landmark_points = [
                        (int(landmark.x * width), int(landmark.y * height))
                        for landmark in landmarks.landmark
                    ]

                    # Define outer face indices
                    outer_face_indices = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                        132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                    ]
                    face_boundary_points = np.array([landmark_points[i] for i in outer_face_indices], np.int32)

                    cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Create a convex hull and mask
                    cv2.fillConvexPoly(mask, face_boundary_points, 255)

                    # Apply mask to extract the face region
                    masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

                    # Calculate mean RGB if pixels exist in the face region
                    n_facepixels = np.sum(mask // 255)
                    if n_facepixels > 0:
                        mean_r = np.sum(masked_face[:, :, 2]) / n_facepixels
                        mean_g = np.sum(masked_face[:, :, 1]) / n_facepixels
                        mean_b = np.sum(masked_face[:, :, 0]) / n_facepixels
                        mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))

                # BPM Estimation
                current_time = time.time()
                if current_time >= bpm_estimation_start:
                    if (current_time - start_time) >= 2.0:
                        start_time = current_time  # Reset timer
                        # Process the mean RGB signal for rPPG
                        l = int(fps * 1.6)
                        if mean_rgb.shape[0] > l:
                            rPPG_signals = POS(mean_rgb, l)
                            rPPG_filtered = bandpass_filter(rPPG_signals, fps)
                            rPPG_filtered = standardization_signal(rPPG_filtered)
                            rPPG_bpm = BPM_estimation(rPPG_filtered, n_segment, fps)
                            bpm_display = f"BPM: {rPPG_bpm:.2f}"
                        else:
                            bpm_display = "BPM: N/A"

                # Update placeholders
                frame_placeholder.image(resized_frame, channels="BGR", use_column_width=True)
                bpm_placeholder.subheader(bpm_display)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            cap.release()
else:
    st.write("Check the box above to start the webcam.")
