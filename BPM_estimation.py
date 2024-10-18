from scipy.signal import welch
import numpy as np


def BPM_estimation(rPPG_filtered, n_segment, fps):
    seg_len = (2 * rPPG_filtered.shape[0]) // n_segment + 1
    freq_rPPG, psd_rPPG = welch(rPPG_filtered, fs=fps, nperseg=seg_len, window='flattop')
    max_freq_rPPG = freq_rPPG[np.argmax(psd_rPPG)]
    rPPG_bpm = max_freq_rPPG * 60
    return(rPPG_bpm)