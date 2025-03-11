import cv2
import numpy as np
import time
import csv
from scipy import signal
from scipy.signal import welch
from signal_processing import *
from BPM_estimation import *
from ROI_selection import *
from POS import *
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
fps = 30
n_segment = 2
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
# # Output CSV File
# csv_filename = "rPPG_data.csv"
# csv_file = open(csv_filename, mode='w', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Timestamp", "BPM"])  # CSV Header

f_cnt = 0
mask = None
n_skinpixels = 0
face_detected = False
delay = 30
bpm_display = "BPM: N/A" 
start_time = time.time()
bpm_estimation_start = start_time + delay  
mean_rgb = np.empty((0, 3)) 
# rPPG_plot_data = np.zeros(150)
forehead_indices = [107, 66, 69, 109, 10, 338, 299, 296, 336, 9]
left_cheek_indices = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
right_cheek_indices = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]
cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Feed', 600, 400) 
cv2.moveWindow('Webcam Feed', 200, 50)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture video.")
            break
        width = int(frame_width * 0.6)
        height = int(frame_height * 0.6)
        resized_frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            if not face_detected:
                face_detected = True
                bpm_estimation_start = time.time() + delay
                bpm_display = "Recording Signal"
            landmarks = results.multi_face_landmarks[0]
            draw_box(resized_frame, height, width, landmarks)
            forehead_mask = extract_roi(forehead_indices, width, height, landmarks)
            left_cheek_mask = extract_roi(left_cheek_indices, width, height, landmarks)
            right_cheek_mask = extract_roi(right_cheek_indices, width, height, landmarks)
            
            combined_mask = cv2.bitwise_or(cv2.bitwise_or(forehead_mask, left_cheek_mask), right_cheek_mask).astype(np.uint8)

            if combined_mask.shape[:2] != resized_frame.shape[:2]:
                combined_mask = cv2.resize(combined_mask, (resized_frame.shape[1], resized_frame.shape[0]))

            masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=combined_mask)
            n_facepixels = np.sum(combined_mask // 255)
            if n_facepixels > 0:
                mean_r = np.sum(masked_face[:, :, 2]) / n_facepixels
                mean_g = np.sum(masked_face[:, :, 1]) / n_facepixels
                mean_b = np.sum(masked_face[:, :, 0]) / n_facepixels
                if f_cnt == 0:
                    mean_rgb = np.array([mean_r, mean_g, mean_b])
                    print(mean_rgb)
                else:
                    mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))
        else:
            face_detected = False
            mean_rgb = np.empty((0, 3))
            # rPPG_plot_data = np.zeros(150)
            rPPG_standard = []
            bpm_display = "BPM: N/A"
        f_cnt += 1  
        current_time = time.time()
        if face_detected and current_time >= bpm_estimation_start:
            l = int(fps * 1.6)
            if mean_rgb.shape[0] > l:
                rPPG_signals = POS(mean_rgb, l)
                rPPG_filtered = bandpass_filter(rPPG_signals, fps)
                rPPG_standard = standardization_signal(rPPG_filtered)
                # rPPG_plot_data = np.roll(rPPG_plot_data, -1)
                # rPPG_plot_data[-1] = np.mean(rPPG_standard)
            if (current_time - start_time) >= 2.0:
                start_time = current_time 
                rPPG_bpm = BPM_estimation(rPPG_standard, n_segment, fps)
                print(f"BPM: {rPPG_bpm:.2f}")
                bpm_display = f"BPM: {rPPG_bpm:.2f}"
                # csv_writer.writerow([current_time, bpm_display])

        cv2.putText(resized_frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # graph_x = 50  
        # graph_y = height - 50
        # graph_width = 300  # Increased width to spread out the graph
        # graph_height = 100

        # if len(rPPG_plot_data) > 1:
        #     norm_rPPG = rPPG_plot_data - np.min(rPPG_plot_data)  # Shift values to positive
        #     range_val = np.ptp(norm_rPPG)  # Peak-to-peak range (max - min)
        #     if range_val > 0:
        #         norm_rPPG = (norm_rPPG / range_val) * graph_height  # Normalize with peak-to-peak scaling
        #     norm_rPPG = graph_y - norm_rPPG  # Invert Y-axis for OpenCV

        #     skip = max(1, len(norm_rPPG) // 100)  # Reduce plotted points if too dense
        #     for i in range(1, len(norm_rPPG), skip):
        #         x1 = graph_x + int((i - 1) * (graph_width / len(norm_rPPG)))
        #         x2 = graph_x + int(i * (graph_width / len(norm_rPPG)))
        #         y1, y2 = int(norm_rPPG[i - 1]), int(norm_rPPG[i])

        #         cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


        # video_writer.write(frame)
        cv2.imshow('Webcam Feed', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  # Press 'q' to exit the loop
        
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting...")

finally:
    cap.release()
    # video_writer.release()
    # csv_file.close()
    cv2.destroyAllWindows()
    # rPPG_filename = "rPPG_filtered.csv"
    # print(f"Saving rPPG_filtered to '{rPPG_filename}'...")
    # with open(rPPG_filename, mode='w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["rPPG_filtered"])  # Header
    #     # If it's 1D, write each sample in a new row
    #     for val in rPPG_standard:
    #         writer.writerow([val])