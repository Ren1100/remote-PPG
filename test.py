import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import GRGB
import signal_plot
import signal_processing

def real_time_rppg():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    buffer_size = int(fps * 3)  # 10-second window

    frames = []
    # rois = []

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

        for (x, y, w, h) in faces:
            # (x, y, w, h) = face
            # Define ROIs
            forehead_roi = (x + w // 3, y+ h//16, w // 3, h // 6)  # Center forehead
            # left_cheek_roi = (x + w // 8, y + h // 2, w // 4, h // 4)  # Left cheek
            # right_cheek_roi = (x + 5 * w // 8, y + h // 2, w // 4, h // 4)  # Right cheek

            # rois = [forehead_roi]
            forehead_frame = frame[forehead_roi[1]:forehead_roi[1] + forehead_roi[3], 
                                   forehead_roi[0]:forehead_roi[0] + forehead_roi[2]]

            # rois = [forehead_roi, left_cheek_roi, right_cheek_roi]

            # Add the current frame's ROIs to the frame buffer
            frames.append(forehead_frame)

            # Draw rectangles around the detected face and ROIs
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (forehead_roi[0], forehead_roi[1]), (forehead_roi[0] + forehead_roi[2], forehead_roi[1] + forehead_roi[3]), (255, 0, 0), 2)
            # cv2.rectangle(frame, (left_cheek_roi[0], left_cheek_roi[1]), (left_cheek_roi[0] + left_cheek_roi[2], left_cheek_roi[1] + left_cheek_roi[3]), (0, 255, 0), 2)
            # cv2.rectangle(frame, (right_cheek_roi[0], right_cheek_roi[1]), (right_cheek_roi[0] + right_cheek_roi[2], right_cheek_roi[1] + right_cheek_roi[3]), (0, 0, 255), 2)

        # Limit the buffer size to the last 10 seconds
        if len(frames) > buffer_size:
            frames.pop(0)

        # Once we have enough frames, process the rPPG signal
        if len(frames) == buffer_size:
            r_signal, g_signal, b_signal, gr_signal, gb_signal, grgb_signal = GRGB.compute_rppg_from_rois(frames)

            # Apply bandpass filter to the signals
            gr_signal_band = signal_processing.bandpass_filter(gr_signal, fps)
            gb_signal_band = signal_processing.bandpass_filter(gb_signal, fps)
            grgb_signal_band = signal_processing.bandpass_filter(grgb_signal, fps)

            # # Normalize the signals using min-max scaling
            # gr_signal_norm = signal_processing.normalize_signal(gr_signal_band)
            # gb_signal_norm = signal_processing.normalize_signal(gb_signal_band)
            # grgb_signal_norm = signal_processing.normalize_signal(grgb_signal_band)

            # Plot the signals
            signal_plot.plot_signals(r_signal, g_signal, b_signal, gr_signal_band, gb_signal_band, grgb_signal_band, fps)

            # Plot FFT power spectrum of the signals
            signal_plot.plot_fft(gr_signal_band, fps, "GR Signal (G/R)")
            signal_plot.plot_fft(gb_signal_band, fps, "GB Signal (G/B)")
            signal_plot.plot_fft(grgb_signal_band, fps, "GRGB Signal (G/R + G/B)")

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
