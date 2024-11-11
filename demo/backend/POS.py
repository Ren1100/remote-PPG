import numpy as np
import pandas as pd

def POS(mean_rgb, l):
    rPPG_signals = np.zeros(mean_rgb.shape[0])

    for t in range(0, mean_rgb.shape[0] - l):
        C = mean_rgb[t:t + l - 1, :].T
        mean_color = np.mean(C, axis=1)
        diag_mean_color_inv = np.linalg.inv(np.diag(mean_color))
        Cn = np.matmul(diag_mean_color_inv, C)

        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        S = np.matmul(projection_matrix, Cn)

        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)

        epsilon = 1e-8  # Small value to avoid division by zero
        rPPG_signals[t:t + l - 1] = rPPG_signals[t:t + l - 1] + (P - np.mean(P)) / (np.std(P) + epsilon)
    return(rPPG_signals)