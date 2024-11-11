from flask import Flask, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from POS import *
from BPM_estimation import *
from signal_processing import *
import time

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
fps = 30
mean_rgb = np.empty((0, 3))
f_cnt = 0
bpm_display = "BPM: N/A"
start_time = time.time()
delay = 15
bpm_estimation_start = start_time + delay

def generate_frames():
    global mean_rgb, f_cnt, bpm_display, start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale_percent = 60
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            mask = np.zeros((height, width), dtype=np.uint8)
            landmark_points = [(int(l.x * width), int(l.y * height)) for l in landmarks.landmark]

            outer_face_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            face_boundary_points = np.array([landmark_points[i] for i in outer_face_indices], np.int32)
            cv2.polylines(resized_frame, [face_boundary_points], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.fillConvexPoly(mask, face_boundary_points, 255)
            masked_face = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)
            x, y, w, h = cv2.boundingRect(face_boundary_points)
            cropped_face = masked_face[y:y+h, x:x+w]
            ycrcb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2YCrCb)

            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb_face, lower_skin, upper_skin)
            skin_segmented_face = cv2.bitwise_and(cropped_face, cropped_face, mask=skin_mask)

            n_facepixels = np.sum(skin_mask // 255)

            if n_facepixels > 0:
                mean_r = np.sum(skin_segmented_face[:, :, 2]) / n_facepixels
                mean_g = np.sum(skin_segmented_face[:, :, 1]) / n_facepixels
                mean_b = np.sum(skin_segmented_face[:, :, 0]) / n_facepixels

                if f_cnt == 0:
                    mean_rgb = np.array([mean_r, mean_g, mean_b])
                else:
                    mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))

        f_cnt += 1
        current_time = time.time()
        if current_time >= bpm_estimation_start and (current_time - start_time) >= 2.0:
            start_time = current_time
            l = int(fps * 1.6)
            if mean_rgb.shape[0] > l:
                rPPG_signals = POS(mean_rgb, l)
                rPPG_filtered = bandpass_filter(rPPG_signals, fps)
                rPPG_filtered = standardization_signal(rPPG_filtered)
                rPPG_bpm = BPM_estimation(rPPG_filtered, 2, fps)
                bpm_display = f"BPM: {rPPG_bpm:.2f}"

        # Add BPM text on the video frame
        cv2.putText(resized_frame, bpm_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bpm')
def get_bpm():
    return jsonify({"bpm": bpm_display.split(": ")[1]})

if __name__ == '__main__':
    app.run(debug=True)
