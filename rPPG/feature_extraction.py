import cv2
import numpy as np

def extract_features_from_frame(frame):
    green_channel = frame[:, :, 1]
    avg_intensity = green_channel.mean()
    var_intensity = green_channel.var()
    std_intensity = green_channel.std()
    return [avg_intensity, var_intensity, std_intensity]
