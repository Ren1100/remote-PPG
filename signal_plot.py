import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft



def plot_fft(signal, fps, title):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fps)
    fft_values = np.abs(fft(signal))

    # Convert frequency to BPM
    bpm = freqs * 60

    # Filter frequencies and values within the desired BPM range
    valid_indices = (bpm >= 40) & (bpm <= 120)
    bpm_valid = bpm[valid_indices]
    fft_values_valid = fft_values[valid_indices]

    # Find the index of the maximum amplitude within the valid range
    max_idx = np.argmax(fft_values_valid)
    max_bpm = bpm_valid[max_idx]
    max_amplitude = fft_values_valid[max_idx]

    # Plot only the positive frequencies
    plt.figure()
    plt.plot(bpm[:n//2], fft_values[:n//2])

    # Annotate the maximum amplitude within the desired BPM range
    plt.scatter(max_bpm, max_amplitude, color='red', label=f'Max Amplitude: {max_amplitude:.2f} at {max_bpm:.2f} BPM')
    plt.legend()
    
    # Set the x-axis limits from 40 to 120 BPM
    plt.xlim(40, 120)
    
    plt.title(f"FFT Power Spectrum of {title} (BPM)")
    plt.xlabel('BPM')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_signals(r_signal, g_signal, b_signal, gr_signal, gb_signal, grgb_signal, fps):
    time_axis = np.arange(len(gr_signal)) / fps

    plt.figure(figsize=(15, 12))

    # Plot R(t)
    plt.subplot(6, 1, 1)
    plt.plot(time_axis, r_signal, label="R(t)", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("R value")
    plt.title("R(t) Signal")
    plt.grid(True)

    # Plot G(t)
    plt.subplot(6, 1, 2)
    plt.plot(time_axis, g_signal, label="G(t)", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("G value")
    plt.title("G(t) Signal")
    plt.grid(True)

    # Plot B(t)
    plt.subplot(6, 1, 3)
    plt.plot(time_axis, b_signal, label="B(t)", color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("B value")
    plt.title("B(t) Signal")
    plt.grid(True)

    # Plot GR signal
    plt.subplot(6, 1, 4)
    plt.plot(time_axis, gr_signal, label="GR (G/R)", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("GR Signal (G/R)")
    plt.grid(True)

    # Plot GB signal
    plt.subplot(6, 1, 5)
    plt.plot(time_axis, gb_signal, label="GB (G/B)", color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("GB Signal (G/B)")
    plt.grid(True)

    # Plot GRGB signal
    plt.subplot(6, 1, 6)
    plt.plot(time_axis, grgb_signal, label="GRGB (G/R + G/B)", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("GRGB Signal (G/R + G/B)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()