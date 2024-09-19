import cv2
import numpy as np
import dlib
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import time

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
        r = np.mean(frame[:, :, 2])  # Red channel
        g = np.mean(frame[:, :, 1])  # Green channel
        b = np.mean(frame[:, :, 0])  # Blue channel

        grgb_signal = (g / r) + (g / b)
        rppg_signal.append(grgb_signal)

    rppg_signal = np.array(rppg_signal)
    filtered_signal = bandpass_filter(rppg_signal, fps)

    return filtered_signal

# Estimate heart rate using FFT
def estimate_heart_rate(filtered_signal, fps):
    N = len(filtered_signal)
    f_signal = fft(filtered_signal)
    freqs = np.fft.fftfreq(N, 1/fps)

    positive_freqs = freqs[:N // 2]
    f_signal = np.abs(f_signal[:N // 2])

    peak_freq_idx = np.argmax(f_signal)
    peak_freq = positive_freqs[peak_freq_idx]

    heart_rate = peak_freq * 60  # convert Hz to beats per minute (BPM)
    return heart_rate

# Visualize facial landmarks for debugging
def visualize_landmarks(frame, landmarks):
    for n in range(68):  # 68 points for dlib shape predictor
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Small blue circles at each landmark

# Extract cheeks and forehead ROI using facial landmarks
def get_face_landmarks_roi(frame, face, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        landmarks = predictor(gray, face)

        # Visualize landmarks for debugging
        visualize_landmarks(frame, landmarks)

        # Adjust ROI to capture larger regions if needed
        left_cheek = frame[landmarks.part(2).y-20:landmarks.part(4).y+20,
                           landmarks.part(2).x-20:landmarks.part(4).x+20]
        
        right_cheek = frame[landmarks.part(12).y-20:landmarks.part(14).y+20,
                            landmarks.part(12).x-20:landmarks.part(14).x+20]

        forehead = frame[landmarks.part(19).y - 40:landmarks.part(19).y + 20,
                         landmarks.part(19).x - 20:landmarks.part(24).x + 20]

        if left_cheek.size == 0 or right_cheek.size == 0 or forehead.size == 0:
            print("One or more ROIs are empty. Skipping this frame.")
            return None, None, None

        return forehead, left_cheek, right_cheek
    except Exception as e:
        print(f"Error processing landmarks: {e}")
        return None, None, None

# Resize the ROI regions
def resize_rois(forehead, left_cheek, right_cheek, size=(100, 100)):
    try:
        if forehead is None or left_cheek is None or right_cheek is None:
            return None, None, None
        
        forehead_resized = cv2.resize(forehead, size)
        left_cheek_resized = cv2.resize(left_cheek, size)
        right_cheek_resized = cv2.resize(right_cheek, size)
        
        return forehead_resized, left_cheek_resized, right_cheek_resized
    except Exception as e:
        print(f"Error resizing ROIs: {e}")
        return None, None, None

# Main real-time rPPG with OpenCV
def real_time_rppg():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS

    frames = []
    buffer_size = int(fps * 10)  # 10-second window

    # Load Haar cascade for face detection and dlib predictor
    predictor = dlib.shape_predictor("./68_points/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    heart_rate = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)
        if len(faces) == 0:
            print("No face detected. Skipping frame.")
            continue

        for face in faces:
            # Extract ROIs
            forehead_frame, left_cheek_frame, right_cheek_frame = get_face_landmarks_roi(frame, face, predictor)

            # Resize ROIs
            if forehead_frame is not None:
                forehead_resized, left_cheek_resized, right_cheek_resized = resize_rois(forehead_frame, left_cheek_frame, right_cheek_frame)

                if forehead_resized is None or left_cheek_resized is None or right_cheek_resized is None:
                    continue

                combined_frame = np.vstack([forehead_resized, left_cheek_resized, right_cheek_resized])
                frames.append(combined_frame)

                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(frames) > buffer_size:
            frames.pop(0)

        if len(frames) == buffer_size:
            filtered_signal = grgb_rppg(frames, fps)
            heart_rate = estimate_heart_rate(filtered_signal, fps)

        cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_rppg()
