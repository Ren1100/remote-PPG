import matplotlib.pyplot as plt
import os
import re
from glob import glob
import numpy as np
import cv2
import dlib
import datetime as dt
import pandas as pd
from scipy import signal

# List of subjects to process
subject_names = ['01-01/', '01-02/', '01-03/', '01-04/', '01-05/', '01-06/',
                 '02-01/', '02-02/', '02-03/', '02-04/', '02-05/', '02-06/',
                 '03-01/', '03-02/', '03-03/', '03-04/', '03-05/', '03-06/',
                 '04-01/', '04-02/', '04-03/', '04-04/', '04-05/', '04-06/',
                 '05-01/', '05-02/', '05-03/', '05-04/', '05-05/', '05-06/',
                 '06-01/', '06-03/', '06-04/', '06-05/', '06-06/',
                 '07-01/', '07-02/', '07-03/', '07-04/', '07-05/', '07-06/',
                 '08-01/', '08-02/', '08-03/', '08-04/', '08-05/', '08-06/',
                 '09-01/', '09-02/', '09-03/', '09-04/', '09-05/', '09-06/',
                 '10-01/', '10-02/', '10-03/', '10-04/', '10-05/', '10-06/',]  # Add more subjects here
dataset_path = 'PURE/'

fps = 30
n_segment = 2
left_expand_ratio = 0.25
top_expand_ratio = 0.25

detector = dlib.get_frontal_face_detector()  # Face detection

# DataFrame to store BPM results
bpm_results = pd.DataFrame(columns=['Subject', 'BPM'])

for subject_name in subject_names:
    img_frame_path = dataset_path + subject_name + subject_name

    # Load all image files for the subject
    png_list = glob(img_frame_path + '*.png')
    png_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    idx_frame_start = 0
    idx_frame_end = len(png_list) - 1

    f_cnt = 0
    i_cnt = idx_frame_start

    face_left, face_top, face_right, face_bottom = 0, 0, 0, 0
    n_skinpixels = 0
    mean_rgb = []

    while i_cnt >= idx_frame_start and i_cnt <= idx_frame_end:
        frame = cv2.imread(png_list[i_cnt])
        h, w, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply face detection
        if f_cnt == 0:
            rect = detector(gray_frame, 0)
            if len(rect) == 0:
                print(f"No face detected in {png_list[i_cnt]}")
                i_cnt += 1
                continue  # Skip this frame if no face is detected
            rect = rect[0]
            left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
            width = abs(right - left)
            height = abs(bottom - top)
            face_left = int(left - (left_expand_ratio / 2 * width))
            face_top = int(top - (top_expand_ratio / 2 * height))
            face_right = right
            face_bottom = bottom

        face = frame[face_top:face_bottom, face_left:face_right]

        # Convert the face region to hsv color space
        face_YCrCb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        # Define the skin color range in YCrCb
        lower_skin_YCrCb = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin_YCrCb = np.array([255, 173, 127], dtype=np.uint8)
        mask_YCrCb = cv2.inRange(face_YCrCb, lower_skin_YCrCb, upper_skin_YCrCb)
        # n_skinpixels = np.sum(mask_YCrCb)
        # masked_face = cv2.bitwise_and(face, face, mask=mask_YCrCb)

        # Define the skin color range in HSV
        lower_skin_hsv = np.array([0, 50, 0], dtype=np.uint8)  # Adjust as needed
        upper_skin_hsv = np.array([25, 150, 255], dtype=np.uint8)  # Adjust as needed
        mask_hsv = cv2.inRange(face_hsv, lower_skin_hsv, upper_skin_hsv)
        # n_skinpixels = np.sum(mask_hsv)
        # masked_face = cv2.bitwise_and(face, face, mask=mask_hsv)

        mask_HSCrCb = cv2.bitwise_and(mask_YCrCb, mask_hsv)
        n_skinpixels = np.sum(mask_HSCrCb)
        masked_face = cv2.bitwise_and(face, face, mask=mask_HSCrCb)

        # Get the mean RGB value in the skin
        mean_r = np.sum(masked_face[:, :, 2]) / n_skinpixels if n_skinpixels > 0 else 0
        mean_g = np.sum(masked_face[:, :, 1]) / n_skinpixels if n_skinpixels > 0 else 0
        mean_b = np.sum(masked_face[:, :, 0]) / n_skinpixels if n_skinpixels > 0 else 0

        if f_cnt == 0:
            mean_rgb = np.array([mean_r, mean_g, mean_b])
        else:
            mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))

        f_cnt += 1
        i_cnt += 1

    # Process the extracted RGB values to calculate rPPG signals
    # l = int(fps * 1.6)
    # rPPG_signals = np.zeros(mean_rgb.shape[0])
    # for t in range(0, mean_rgb.shape[0] - l):
    #     C = mean_rgb[t:t+l-1, :].T
    #     mean_color = np.mean(C, axis=1)
    #     diag_mean_color_inv = np.linalg.inv(np.diag(mean_color))
    #     Cn = np.matmul(diag_mean_color_inv, C)
    #     projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    #     S = np.matmul(projection_matrix, Cn)
    #     std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
    #     P = np.matmul(std, S)
    #     epsilon = 1e-8
    #     rPPG_signals[t:t+l-1] = rPPG_signals[t:t+l-1] + (P - np.mean(P)) / (np.std(P) + epsilon)
    R = mean_rgb[:, 0]
    G = mean_rgb[:, 1]
    B = mean_rgb[:, 2]

    GRGB_signal = (G/R) + (G/B)

    # # Filter the rPPG signals
    # lowcut = 0.65
    # highcut = 4
    # b, a = signal.butter(6, [lowcut, highcut], btype='bandpass', fs=fps)
    # rPPG_filtered = signal.filtfilt(b, a, rPPG_signals)

    # # Standardize the filtered signals
    # rPPG_filtered = (rPPG_filtered - np.mean(rPPG_filtered)) / np.std(rPPG_filtered)

    # Filter the GRGB signals
    lowcut = 0.65
    highcut = 4
    b, a = signal.butter(6, [lowcut, highcut], btype='bandpass', fs=fps)
    GRGB_filtered = signal.filtfilt(b, a, GRGB_signal)

    #standardization
    GRGB_filtered = (GRGB_filtered-np.mean(GRGB_filtered))/np.std(GRGB_filtered)

    # # Calculate the BPM from the rPPG signals
    # seg_len = (2 * rPPG_filtered.shape[0]) // n_segment + 1
    # freq_rPPG, psd_rPPG = signal.welch(rPPG_filtered, fs=fps, nperseg=seg_len, window='flattop')
    # max_freq_rPPG = freq_rPPG[np.argmax(psd_rPPG)]
    # rPPG_bpm = max_freq_rPPG * 60
    # print(f"Subject {subject_name}: BPM: {rPPG_bpm}")

    # Calculate the BPM from the rPPG signals
    seg_len = (2 * GRGB_filtered.shape[0]) // n_segment + 1
    freq_GRGB, psd_GRGB = signal.welch(GRGB_filtered, fs=fps, nperseg=seg_len, window='flattop')
    max_freq_GRGB = freq_GRGB[np.argmax(psd_GRGB)]
    GRGB_bpm = max_freq_GRGB * 60
    print(f"Subject {subject_name}: BPM: {GRGB_bpm}")

    # Store the BPM result in the DataFrame
    new_row = pd.DataFrame({'Subject': [subject_name], 'BPM': [GRGB_bpm]})
    bpm_results = pd.concat([bpm_results, new_row], ignore_index=True)

# Save BPM results to an Excel file
output_file = 'BPM_results.xlsx'
bpm_results.to_excel(output_file, index=False)
print(f"BPM results saved to {output_file}")
