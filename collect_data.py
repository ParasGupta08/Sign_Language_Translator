# =============================================================================
# collect_data.py — Record gesture clips from your webcam
#
# Usage:
#   python collect_data.py
#
# Controls (during recording window):
#   SPACE  → start recording a clip
#   Q      → quit / move to next sign
# =============================================================================

import cv2
import os
import numpy as np
from config import (
    SIGNS, CLIPS_PER_SIGN, FRAMES_PER_CLIP,
    FRAME_SIZE, DATA_DIR,
    SKIN_HSV_LOWER, SKIN_HSV_UPPER, ROI_PADDING,
    WEBCAM_INDEX
)


# ─────────────────────────────────────────
# Utility: isolate hand region via skin mask
# ─────────────────────────────────────────
def extract_hand_roi(frame):
    """
    Hand detection using a fixed right-side zone.
    Instead of fighting the skin mask over face/neck,
    we only look in the RIGHT 55% of the frame where
    the user holds their hand. Face is always on the left/centre.
    """
    fh, fw = frame.shape[:2]

    # ── Only search in right portion of frame ──
    # User holds hand on the right side; face is centre-left
    zone_x_start = int(fw * 0.35)   # start at 35% from left
    zone = frame[:, zone_x_start:]  # right 65% of frame
    zh, zw = zone.shape[:2]

    # ── HSV skin mask on zone only ──
    hsv   = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   15, 60],  dtype=np.uint8),
                              np.array([25, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([165, 15,  60],  dtype=np.uint8),
                              np.array([180, 255, 255], dtype=np.uint8))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    # ── Find largest contour in zone ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 1500:
        return None, None

    # ── Square crop centred on contour ──
    x, y, w, h = cv2.boundingRect(best)

    # translate back to full frame coordinates
    x += zone_x_start

    min_side = int(min(fw, fh) * 0.20)
    max_side = int(min(fw, fh) * 0.55)
    side     = max(min_side, min(max(w, h), max_side))

    cx   = x + w // 2
    cy   = y + h // 2
    half = side // 2 + ROI_PADDING

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(fw, cx + half)
    y2 = min(fh, cy + half)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, FRAME_SIZE)

    return roi_gray, (x1, y1, x2 - x1, y2 - y1)


# ─────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────
def collect():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check WEBCAM_INDEX in config.py.")

    print("\n" + "="*55)
    print("  SIGN LANGUAGE DATA COLLECTION")
    print("="*55)
    print(f"  Signs to record : {SIGNS}")
    print(f"  Clips per sign  : {CLIPS_PER_SIGN}")
    print(f"  Frames per clip : {FRAMES_PER_CLIP}")
    print("="*55)
    print("  SPACE = start clip   |   Q = skip / quit sign\n")

    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign.replace(" ", "_"))
        os.makedirs(sign_dir, exist_ok=True)

        # count already collected clips so we can resume
        existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        clip_idx  = existing

        print(f"\n>>> Sign: [{sign}]  ({existing}/{CLIPS_PER_SIGN} already collected)")

        while clip_idx < CLIPS_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)           # mirror for natural feel
            display = frame.copy()

            roi_gray, bbox = extract_hand_roi(frame)

            # draw hand bbox if detected
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # HUD
            cv2.putText(display, f"Sign: {sign}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, f"Clip: {clip_idx}/{CLIPS_PER_SIGN}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "SPACE=record  Q=next sign", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print(f"  Skipping remaining clips for [{sign}].")
                break

            if key == ord(' '):
                # ── record one clip ──
                frames_list = []
                print(f"  Recording clip {clip_idx + 1}...", end=" ", flush=True)

                while len(frames_list) < FRAMES_PER_CLIP:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame    = cv2.flip(frame, 1)
                    roi_gray, bbox = extract_hand_roi(frame)

                    if roi_gray is not None:
                        frames_list.append(roi_gray)

                    # live preview during recording
                    display2 = frame.copy()
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(display2, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(display2, f"RECORDING {len(frames_list)}/{FRAMES_PER_CLIP}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Data Collection", display2)
                    cv2.waitKey(1)

                if len(frames_list) == FRAMES_PER_CLIP:
                    clip_array = np.stack(frames_list, axis=0)   # (T, H, W)
                    save_path  = os.path.join(sign_dir, f"clip_{clip_idx:04d}.npy")
                    np.save(save_path, clip_array)
                    clip_idx += 1
                    print(f"saved → {save_path}")
                else:
                    print("incomplete clip (hand not detected enough). Try again.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Data collection complete.")


if __name__ == "__main__":
    collect()