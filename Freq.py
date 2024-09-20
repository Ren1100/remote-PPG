import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# Bandpass filter (0.65 to 4 Hz)
def bandpass_filter(signal, fs, lowcut=0.65, highcut=3.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Normalization function using min-max scaling
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(signal)

# Compute rPPG signals using multiple ROIs
def compute_rppg_from_rois(frames, rois):
    gr_signal, gb_signal, grgb_signal = [], [], []

    for frame in frames:
        r_total, g_total, b_total = 0, 0, 0
        for roi in rois:
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            
            # Calculate mean RGB values for the current ROI
            r = np.mean(roi_frame[:, :, 2])  # Red channel
            g = np.mean(roi_frame[:, :, 1])  # Green channel
            b = np.mean(roi_frame[:, :, 0])  # Blue channel
            
            r_total += r
            g_total += g
            b_total += b
        
        # Average RGB values from all ROIs
        num_rois = len(rois)
        r_avg = r_total / num_rois
        g_avg = g_total / num_rois
        b_avg = b_total / num_rois
        
        # Calculate GR, GB, and GRGB signals
        gr = g_avg / r_avg if r_avg != 0 else 0
        gb = g_avg / b_avg if b_avg != 0 else 0
        grgb = (g_avg / r_avg) + (g_avg / b_avg) if r_avg != 0 and b_avg != 0 else 0

        gr_signal.append(gr)
        gb_signal.append(gb)
        grgb_signal.append(grgb)

    return np.array(gr_signal), np.array(gb_signal), np.array(grgb_signal)

def compute_power_spectrum(signal, fs):
    # Compute the Fourier Transform of the signal
    N = len(signal)
    yf = fft(signal)
    xf = np.fft.fftfreq(N, 1/fs)
    
    # Compute the power spectrum
    power_spectrum = np.abs(yf)**2 / N
    xf = xf[:N//2]  # Only take the positive frequencies
    power_spectrum = power_spectrum[:N//2]
    
    return xf, power_spectrum

def plot_power_spectrum(xf, power_spectrum, signal_name):
    plt.figure(figsize=(10, 4))
    plt.plot(xf, power_spectrum, label=f'Power Spectrum of {signal_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(f'Power Spectrum of {signal_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

def real_time_rppg():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    buffer_size = int(fps * 10)  # 10-second window

    frames = []
    rois = []

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
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
            (x, y, w, h) = face
            # Define ROIs
            forehead_roi = (x + w // 4, y, w // 2, h // 4)  # Center forehead
            left_cheek_roi = (x + w // 8, y + h // 2, w // 4, h // 4)  # Left cheek
            right_cheek_roi = (x + 5 * w // 8, y + h // 2, w // 4, h // 4)  # Right cheek

            rois = [forehead_roi, left_cheek_roi, right_cheek_roi]

            # Add the current frame's ROIs to the frame buffer
            frames.append(frame)

            # Draw rectangles around the detected face and ROIs
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (forehead_roi[0], forehead_roi[1]), (forehead_roi[0] + forehead_roi[2], forehead_roi[1] + forehead_roi[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (left_cheek_roi[0], left_cheek_roi[1]), (left_cheek_roi[0] + left_cheek_roi[2], left_cheek_roi[1] + left_cheek_roi[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_cheek_roi[0], right_cheek_roi[1]), (right_cheek_roi[0] + right_cheek_roi[2], right_cheek_roi[1] + right_cheek_roi[3]), (0, 0, 255), 2)

        # Limit the buffer size to the last 10 seconds
        if len(frames) > buffer_size:
            frames.pop(0)

        # Once we have enough frames, process the rPPG signal
        if len(frames) == buffer_size:
            gr_signal, gb_signal, grgb_signal = compute_rppg_from_rois(frames, rois)

            # Apply bandpass filter to the signals
            gr_signal = bandpass_filter(gr_signal, fps)
            gb_signal = bandpass_filter(gb_signal, fps)
            grgb_signal = bandpass_filter(grgb_signal, fps)

            # Normalize the signals using min-max scaling
            gr_signal = normalize_signal(gr_signal)
            gb_signal = normalize_signal(gb_signal)
            grgb_signal = normalize_signal(grgb_signal)

           # Perform power spectrum analysis
            xf_gr, power_spectrum_gr = compute_power_spectrum(gr_signal, fps)
            xf_gb, power_spectrum_gb = compute_power_spectrum(gb_signal, fps)
            xf_grgb, power_spectrum_grgb = compute_power_spectrum(grgb_signal, fps)

            # Plot the power spectra
            plot_power_spectrum(xf_gr, power_spectrum_gr, "GR Signal")
            plot_power_spectrum(xf_gb, power_spectrum_gb, "GB Signal")
            plot_power_spectrum(xf_grgb, power_spectrum_grgb, "GRGB Signal")

            # Reset frame buffer after processing
            frames = []

        # Display the video feed
        cv2.imshow("Webcam", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_rppg()
