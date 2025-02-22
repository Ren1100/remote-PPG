import cv2
import numpy as np
import time
import csv
from scipy import signal
from scipy.signal import welch
from signal_processing import *
from BPM_estimation import *
from POS import *
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
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
csv_writer.writerow(["Timestamp", "BPM"])  # CSV Header

f_cnt = 0
face_left, face_top, face_right, face_bottom = 0, 0, 0, 0
mask = None
n_skinpixels = 0
face_detected = False
delay = 10
bpm_display = "BPM: N/A" 
start_time = time.time()
bpm_estimation_start = start_time + delay  
mean_rgb = np.empty((0, 3)) 
rPPG_plot_data = np.zeros(90)
cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
cv2.moveWindow('Webcam Feed', 500, 100)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture video.")
            break
        scale_percent = 60 
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            if not face_detected:
                face_detected = True
                bpm_estimation_start = time.time() + delay
            landmarks = results.multi_face_landmarks[0]
            mask = np.zeros((height, width), dtype=np.uint8)
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))
            outer_face_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            face_boundary_points = np.array([landmark_points[i] for i in outer_face_indices], np.int32)
            cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillConvexPoly(mask, face_boundary_points, 255)
            kernel = np.ones((1, 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)
            ycrcb_face = cv2.cvtColor(masked_face, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb_face, lower_skin, upper_skin)
            skin_segmented_face = cv2.bitwise_and(masked_face, masked_face, mask=skin_mask)
            n_facepixels = np.sum(skin_mask // 255)
            if n_facepixels > 0:
                mean_r = np.sum(skin_segmented_face[:, :, 2]) / n_facepixels
                mean_g = np.sum(skin_segmented_face[:, :, 1]) / n_facepixels
                mean_b = np.sum(skin_segmented_face[:, :, 0]) / n_facepixels
                if f_cnt == 0:
                    mean_rgb = np.array([mean_r, mean_g, mean_b])
                    print(mean_rgb)
                else:
                    mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))
        else:
            face_detected = False
            mean_rgb = np.empty((0, 3))
            rPPG_plot_data = np.zeros(90)
            rPPG_filtered = []
            bpm_display = "BPM: N/A"
        f_cnt += 1

        current_time = time.time()
        if face_detected and current_time >= bpm_estimation_start:
            l = int(fps * 1.6)
            if mean_rgb.shape[0] > l:
                rPPG_signals = POS(mean_rgb, l)
                rPPG_filtered = bandpass_filter(rPPG_signals, fps)
                rPPG_filtered = standardization_signal(rPPG_filtered)
                rPPG_plot_data = np.roll(rPPG_plot_data, -1)
                rPPG_plot_data[-1] = np.mean(rPPG_filtered)
            if (current_time - start_time) >= 2.0:
                start_time = current_time 
                rPPG_bpm = BPM_estimation(rPPG_filtered, n_segment, fps)
                print(f"BPM: {rPPG_bpm:.2f}")
                bpm_display = f"BPM: {rPPG_bpm:.2f}"
                csv_writer.writerow([current_time, bpm_display])

        cv2.putText(resized_frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        graph_x = 50  
        graph_y = height - 100
        graph_width = 150
        graph_height = 80 
        if len(rPPG_plot_data) > 1:
            norm_rPPG = rPPG_plot_data - np.min(rPPG_plot_data)  # Shift values to positive
            norm_rPPG = (norm_rPPG / (np.max(norm_rPPG) + 1e-6)) * graph_height  # Normalize and scale
            norm_rPPG = graph_y - norm_rPPG  # Invert Y-axis for OpenCV

            for i in range(1, len(norm_rPPG)):
                x1 = graph_x + int((i - 1) * (graph_width / len(norm_rPPG)))
                x2 = graph_x + int(i * (graph_width / len(norm_rPPG)))
                y1, y2 = int(norm_rPPG[i - 1]), int(norm_rPPG[i])

                cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # video_writer.write(frame)
        cv2.imshow('Webcam Feed', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  # Press 'q' to exit the loop
        
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting...")

finally:
    cap.release()
    video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    rPPG_filename = "rPPG_filtered.csv"
    print(f"Saving rPPG_filtered to '{rPPG_filename}'...")
    with open(rPPG_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rPPG_filtered"])  # Header
        # If it's 1D, write each sample in a new row
        for val in rPPG_filtered:
            writer.writerow([val])