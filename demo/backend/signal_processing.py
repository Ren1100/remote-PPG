import numpy as np
from scipy import signal


def bandpass_filter(rPPG_signals, fps):
    lowcut = 0.65
    highcut = 4
    b, a = signal.butter(6, [lowcut, highcut], btype='bandpass', fs=fps)
    rPPG_filtered = signal.filtfilt(b, a, rPPG_signals)
    return (rPPG_filtered)

# Normalization function using min-max scaling
def standardization_signal(rPPG_filtered):
    rPPG_filtered = (rPPG_filtered - np.mean(rPPG_filtered)) / np.std(rPPG_filtered)
    return (rPPG_filtered)