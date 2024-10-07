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

# Define time duration for recording (in seconds)
record_duration = 5
start_time = time.time()

# Frame count
f_cnt = 0

while (time.time() - start_time) < record_duration:

    ret, frame = cap.read()

    h, w, _ = frame.shape

    # frame_toshow = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame_toshow)
    # print(f"Height: {h}, WidthL {w}")

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

    # frame_toshow = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame_toshow)

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

    # frame_toshow= cv2.cvtColor(masked_face, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame_toshow)

    #Get the mean RGB value in the skin
    mean_r = np.sum(masked_face[:,:,2]) / n_skinpixels
    mean_g = np.sum(masked_face[:,:,1]) / n_skinpixels
    mean_b = np.sum(masked_face[:,:,0]) / n_skinpixels

    if f_cnt ==0:
        mean_rgb = np.array([mean_r, mean_g, mean_b])
    else:
        mean_rgb = np.vstack((mean_rgb,np.array([mean_r, mean_g, mean_b])))

    # cv2.imshow('Webcam Feed', frame)

    f_cnt += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# # Print the collected RGB array
# plt.figure(figsize=(20,5))
# plt.plot(mean_rgb[:,0], label = 'R', color='red')
# plt.plot(mean_rgb[:,1], label = 'G', color='green')
# plt.plot(mean_rgb[:,2], label = 'B', color='blue')
# plt.legend()
# plt.title("Mean RGB")
# plt.show()

l = int(fps * 1.6)
rPPG_signals = np.zeros(mean_rgb.shape[0])

for t in range (0, mean_rgb.shape[0] - l):

    ## CREATE C DATA
    C = mean_rgb[t:t+l-1,:].T

    # print(f"C shape: {C.shape}")
    # print (f"C: \n{C}")

    ## CREATE TEMPORAL NORMALIZATION

    mean_color = np.mean(C, axis=1)

    # print(f"Shape of mean color: {mean_color.shape}")
    # print(f"Mean Color: {mean_color}")

    diag_mean_color = np.diag(mean_color)

    # print(f"Shape of diag mean color: {diag_mean_color.shape}")
    # print(f"Diag Mean Color: \n{diag_mean_color}")

    diag_mean_color_inv = np.linalg.inv(diag_mean_color)
    # print(f"Shape of diag mean color inv: {diag_mean_color_inv.shape}")
    # print(f"Diag Mean Color Inv: \n{diag_mean_color_inv}")

    Cn = np.matmul(diag_mean_color_inv, C)
    # print(f"Cn Shape: {Cn.shape}")



    ##FROM 3D TO 2D

    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]]) 

    S= np.matmul(projection_matrix, Cn)
    # print(f"Shape of S: {S.shape}")

    ##FROM 2D TO 1D

    std = np.array([1, np.std(S[0,:]) / np.std(S[1,:])])
    # print(f"Shape of std: {std.shape}")
    # print(f"std: {std}")

    P = np.matmul(std, S)
    # print(f"Shape of P: {P.shape}")

    # plt.figure(figsize=(20,5))
    # plt.plot(P)

    ## OVERLAPPING

    epsilon = 1e-8  # Small value to avoid division by zero
    rPPG_signals[t:t+l-1] = rPPG_signals[t:t+l-1] + (P - np.mean(P)) / (np.std(P) + epsilon)



    # break

lowcut = 0.65
highcut = 4

b, a = signal.butter(6, [lowcut, highcut], btype='bandpass', fs=fps)
rPPG_filtered = signal.filtfilt(b, a, rPPG_signals)

#standardization
rPPG_filtered = (rPPG_filtered-np.mean(rPPG_filtered))/np.std(rPPG_filtered)

seg_len = (2*rPPG_filtered.shape[0]) // n_segment + 1

freq_rPPG, psd_rPPG = welch(rPPG_filtered, fs=fps, nperseg=seg_len, window='flattop')

max_freq_rPPG =  freq_rPPG[np.argmax(psd_rPPG)]

rPPG_bpm = max_freq_rPPG * 60
print(f"BPM: {rPPG_bpm}")

