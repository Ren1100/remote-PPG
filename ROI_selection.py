import cv2
import numpy as np
def draw_box(resized_frame, height, width, landmarks):

    face_points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks.landmark]
        
    if face_points:
        x_min = min(pt[0] for pt in face_points)
        y_min = min(pt[1] for pt in face_points)
        x_max = max(pt[0] for pt in face_points)
        y_max = max(pt[1] for pt in face_points)

        # Draw bounding box around the detected face
        bbox_points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        cv2.polylines(resized_frame, [np.array(bbox_points)], isClosed=True, color=(0, 255, 0), thickness=2)
def extract_roi(landmark_indices, height, width, landmarks):
    mask = np.zeros((height, width), dtype=np.uint8)
    landmark_points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        landmark_points.append((x, y))
    face_boundary_points = np.array([landmark_points[i] for i in landmark_indices], np.int32)
    # cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.fillConvexPoly(mask, face_boundary_points, 255)
    return mask