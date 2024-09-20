import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.65, highcut=4.0, order=5):
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