import numpy as np

def compute_rppg_from_rois(frames):
    gr_signal, gb_signal, grgb_signal = [], [], []
    r_signal, g_signal, b_signal = [], [], []

    for frame in frames:
        # r_total, g_total, b_total = 0, 0, 0
        # for roi in rois:
        #     x, y, w, h = roi
        #     roi_frame = frame[y:y+h, x:x+w]
            
        #     # Calculate mean RGB values for the current ROI
        #     r = np.mean(roi_frame[:, :, 2])  # Red channel
        #     g = np.mean(roi_frame[:, :, 1])  # Green channel
        #     b = np.mean(roi_frame[:, :, 0])  # Blue channel
            
        #     r_total += r
        #     g_total += g
        #     b_total += b
        
        # # Average RGB values from all ROIs

        r = np.mean(frame[:, :, 2])  # Red channel
        g = np.mean(frame[:, :, 1])  # Green channel
        b = np.mean(frame[:, :, 0])  # Blue channel

        # num_rois = len(rois)
        # r_avg = r_total / num_rois
        # g_avg = g_total / num_rois
        # b_avg = b_total / num_rois
        
        # Calculate GR, GB, and GRGB signals
        gr = g / r
        gb = g / b 
        grgb = (g / r) + (g / b) 

        r_signal.append(r)
        g_signal.append(g)
        b_signal.append(b)

        # print(f"R(t): {r_signal}")
        # print(f"G(t): {g_signal}")
        # print(f"B(t): {b_signal}")

        gr_signal.append(gr)
        gb_signal.append(gb)
        grgb_signal.append(grgb)

        # print(f"GR(t): {gr_signal}")
        # print(f"GB(t): {gb_signal}")
        # print(f"GRGB(t): {grgb_signal}")

    return np.array(gr_signal), np.array(gb_signal), np.array(grgb_signal), np.array(r_signal), np.array(g_signal), np.array(b_signal)