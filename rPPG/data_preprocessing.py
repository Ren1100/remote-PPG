import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(data_dir, window_size):
    X = []
    y = []
    for session in os.listdir(data_dir):
        session_dir = os.path.join(data_dir, session)
        if not os.path.isdir(session_dir):
            continue
        frame_dir = os.path.join(session_dir, session)
        json_file = os.path.join(session_dir, session + '.json')
        
        if not os.path.exists(json_file):
            continue
        
        features = extract_features_from_frames(frame_dir)
        
        with open(json_file, 'r') as f:
            hr_data = json.load(f)
            heart_rate = np.array(hr_data['heart_rate'])
        
        if len(features) >= window_size:
            for i in range(len(features) - window_size + 1):
                window_features = features[i:i + window_size]
                X.append(window_features.flatten())
                y.append(np.mean(heart_rate))  # You might adjust this depending on your data
            
    X = np.array(X)
    y = np.array(y)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y
