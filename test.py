import cv2
import numpy as np
import dlib
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
fps = 30
n_segment = 2
left_expand_ratio = 0.25
top_expand_ratio = 0.25

f_cnt = 0

face_left, face_top, face_right, face_bottom = 0,0,0,0
mask = None
n_skinpixels = 0

delay = 10
bpm_display = "BPM: N/A"  # Initialize BPM display text

# Define time duration for recording (in seconds)
start_time = time.time()
bpm_estimation_start = start_time + delay  # Set the initial BPM estimation time

# Frame count
f_cnt = 0

while True:

    ret, frame = cap.read()

    h, w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #apply face detection
    if f_cnt == 0:
        rect = detector(gray_frame, 0)
        rect = rect[0]
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

        width = abs (right - left)
        height = abs (bottom - top)
        face_left = int(left - (left_expand_ratio/2 * width))
        face_top = int (top - (top_expand_ratio/2 * height ))
        face_right = right
        face_bottom = bottom
    
    cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)

    face = frame[face_top:face_bottom, face_left:face_right]

    # Convert the face region to YCrCb color space
    face_YCrCb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    # Define the skin color range in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Create a binary mask where skin color is white and the rest is black
    mask = cv2.inRange(face_YCrCb, lower_skin, upper_skin)
    n_skinpixels = np.sum(mask)

    # Apply the mask to the face region
    masked_face = cv2.bitwise_and(face, face, mask=mask)

    #Get the mean RGB value in the skin
    mean_r = np.sum(masked_face[:,:,2]) / n_skinpixels
    mean_g = np.sum(masked_face[:,:,1]) / n_skinpixels
    mean_b = np.sum(masked_face[:,:,0]) / n_skinpixels

    if f_cnt ==0:
        mean_rgb = np.array([mean_r, mean_g, mean_b])
    else:
        mean_rgb = np.vstack((mean_rgb,np.array([mean_r, mean_g, mean_b])))

    f_cnt += 1
    
    current_time = time.time()
    if current_time >= bpm_estimation_start:
        if (current_time - start_time) >= 2.0:
            start_time = current_time  # Reset timer

            # Process the mean RGB signal for rPPG
            l = int(fps * 1.6)
            if mean_rgb.shape[0] > l:
                rPPG_signals = np.zeros(mean_rgb.shape[0])

                for t in range(0, mean_rgb.shape[0] - l):
                    C = mean_rgb[t:t+l-1, :].T
                    mean_color = np.mean(C, axis=1)
                    diag_mean_color_inv = np.linalg.inv(np.diag(mean_color))
                    Cn = np.matmul(diag_mean_color_inv, C)

                    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
                    S = np.matmul(projection_matrix, Cn)

                    std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
                    P = np.matmul(std, S)

                    epsilon = 1e-8  # Small value to avoid division by zero
                    rPPG_signals[t:t+l-1] = rPPG_signals[t:t+l-1] + (P - np.mean(P)) / (np.std(P) + epsilon)

                # Apply bandpass filter
                lowcut = 0.65
                highcut = 4
                b, a = signal.butter(6, [lowcut, highcut], btype='bandpass', fs=fps)
                rPPG_filtered = signal.filtfilt(b, a, rPPG_signals)

                # Standardization
                rPPG_filtered = (rPPG_filtered - np.mean(rPPG_filtered)) / np.std(rPPG_filtered)

                # Welch's method to estimate frequency
                seg_len = (2 * rPPG_filtered.shape[0]) // n_segment + 1
                freq_rPPG, psd_rPPG = welch(rPPG_filtered, fs=fps, nperseg=seg_len, window='flattop')

                max_freq_rPPG = freq_rPPG[np.argmax(psd_rPPG)]
                rPPG_bpm = max_freq_rPPG * 60
                print(f"BPM: {rPPG_bpm:.2f}")
                bpm_display = f"BPM: {rPPG_bpm:.2f}"  # Update the BPM display text

    
    cv2.putText(frame, bpm_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

