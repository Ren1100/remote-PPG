import cv2
import numpy as np

def YCrCB_Segment(face):
# Convert the face region to YCrCb color space
    face_YCrCb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    # Define the skin color range in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Create a binary mask where skin color is white and the rest is black
    mask = cv2.inRange(face_YCrCb, lower_skin, upper_skin)
    n_skinpixels = np.sum(mask)

    if n_skinpixels == 0:
        print("No skin pixels detected.")
        return None, None

    # Apply the mask to the face region
    masked_face = cv2.bitwise_and(face, face, mask=mask)
    return (masked_face, n_skinpixels)