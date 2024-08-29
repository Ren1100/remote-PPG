import cv2
import numpy as np

def extract_features_from_frames(frame_dir):
    feature_list = []
    frame_files = sorted(os.listdir(frame_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        # Extract green channel
        green_channel = frame[:, :, 1]
        # Calculate various statistics
        avg_intensity = green_channel.mean()
        var_intensity = green_channel.var()
        std_intensity = green_channel.std()
        feature_list.append([avg_intensity, var_intensity, std_intensity])
    return np.array(feature_list)
