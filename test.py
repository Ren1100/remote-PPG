import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft

# Bandpass filter (0.65 to 4 Hz)
def bandpass_filter(signal, fs, lowcut=0.65, highcut=4.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# GRGB method for rPPG
def grgb_rppg(frames, fps):
    rppg_signal = []

    for frame in frames:
        # Extract RGB channels (mean of each channel)
        r = np.mean(frame[:, :, 2])  # Red channel
        g = np.mean(frame[:, :, 1])  # Green channel
        b = np.mean(frame[:, :, 0])  # Blue channel

        # Calculate GRGB signal as (G/R) + (G/B)
        grgb_signal = (g / r) + (g / b)

        # Append to signal list
        rppg_signal.append(grgb_signal)

    # Convert list to array
    rppg_signal = np.array(rppg_signal)

    # Apply bandpass filter
    filtered_signal = bandpass_filter(rppg_signal, fps)

    return filtered_signal

# Estimate heart rate using FFT
def estimate_heart_rate(filtered_signal, fps):
    N = len(filtered_signal)
    f_signal = fft(filtered_signal)
    freqs = np.fft.fftfreq(N, 1/fps)

    # Only look at positive frequencies
    positive_freqs = freqs[:N // 2]
    f_signal = np.abs(f_signal[:N // 2])

    # Find the peak frequency
    peak_freq_idx = np.argmax(f_signal)
    peak_freq = positive_freqs[peak_freq_idx]

    # Convert peak frequency to BPM
    heart_rate = peak_freq * 60  # convert Hz to beats per minute (BPM)
    return heart_rate

# Function to extract specific ROIs
def get_rois(frame, face):
    x, y, w, h = face
    roi_size = (100, 100)  # Fixed size for resizing

    # More focused ROI areas for forehead and cheeks
    forehead_roi = frame[y:y + h//4, x + w//4:x + 3*w//4]  # Center forehead
    left_cheek_roi = frame[y + h//2:y + 3*h//4, x + w//8:x + 3*w//8]  # Left cheek
    right_cheek_roi = frame[y + h//2:y + 3*h//4, x + 5*w//8:x + 7*w//8]  # Right cheek

    # Resize the regions to ensure consistency
    forehead_roi = cv2.resize(forehead_roi, roi_size)
    left_cheek_roi = cv2.resize(left_cheek_roi, roi_size)
    right_cheek_roi = cv2.resize(right_cheek_roi, roi_size)

    return forehead_roi, left_cheek_roi, right_cheek_roi

def real_time_rppg():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    buffer_size = int(fps * 10)  # 10-second window

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    last_print_time = start_time
    heart_rate = 0

    roi_size = (100, 100)  # Define a fixed size for resizing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        except Exception as e:
            continue

        for face in faces:
            # Extract ROIs
            forehead_roi, left_cheek_roi, right_cheek_roi = get_rois(frame, face)

            # Display ROIs
            cv2.imshow("Forehead ROI", forehead_roi)
            cv2.imshow("Left Cheek ROI", left_cheek_roi)
            cv2.imshow("Right Cheek ROI", right_cheek_roi)

            # Draw rectangle around the detected face
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw bounding boxes around the detected ROIs
            cv2.rectangle(frame, (x + w//4, y), (x + 3*w//4, y + h//4), (255, 0, 0), 2)  # Forehead
            cv2.rectangle(frame, (x + w//8, y + h//2), (x + 3*w//8, y + 3*h//4), (0, 255, 0), 2)  # Left cheek
            cv2.rectangle(frame, (x + 5*w//8, y + h//2), (x + 7*w//8, y + 3*h//4), (0, 0, 255), 2)  # Right cheek

        # Keep buffer size
        if len(frames) > buffer_size:
            frames.pop(0)

        # Perform heart rate detection every second
        if len(frames) == buffer_size:
            filtered_signal = grgb_rppg(frames, fps)
            heart_rate = estimate_heart_rate(filtered_signal, fps)
            start_time = time.time()  # Reset the start time after heart rate calculation

        # Display heart rate on the video feed
        cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Print heart rate every 10 seconds
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
            last_print_time = current_time  # Update last print time

        # Display the video
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_rppg()
