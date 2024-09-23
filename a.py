import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def extract_rppg_from_frame(frame):
    roi = frame[100:200, 100:200]  # Example ROI, adjust as needed
    r, g, b = cv2.split(roi)
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    return r_mean, g_mean, b_mean

def main():
    cap = cv2.VideoCapture(0)  # Use the default camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    r_values, g_values, b_values = deque(maxlen=int(fps*10)), deque(maxlen=int(fps*10)), deque(maxlen=int(fps*10))

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, fps*10)
    ax.grid()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r_mean, g_mean, b_mean = extract_rppg_from_frame(frame)
        r_values.append(r_mean)
        g_values.append(g_mean)
        b_values.append(b_mean)

        if len(r_values) == fps * 10:
            r_values_np = np.array(r_values)
            g_values_np = np.array(g_values)
            b_values_np = np.array(b_values)

            gr = g_values_np / r_values_np
            gb = g_values_np / b_values_np
            grgb = gr + gb

            filtered_grgb = butter_bandpass_filter(grgb, 0.65, 4, fps)

            line.set_ydata(filtered_grgb)
            line.set_xdata(np.arange(len(filtered_grgb)))
            ax.draw_artist(ax.patch)
            ax.draw_artist(line)
            fig.canvas.flush_events()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
