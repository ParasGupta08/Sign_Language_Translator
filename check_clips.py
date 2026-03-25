# paste this as check_clips.py in your project folder
import numpy as np
import cv2
import os
from config import DATA_DIR, SIGNS

def view_clips():
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign.replace(" ", "_"))
        if not os.path.isdir(sign_dir):
            print(f"No folder found for {sign}")
            continue
        
        clips = sorted([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        print(f"\n[{sign}] — {len(clips)} clips found")
        
        if not clips:
            continue

        # show first clip of each sign
        clip = np.load(os.path.join(sign_dir, clips[0]))  # (T, H, W)
        print(f"  Shape: {clip.shape}")

        for frame in clip:
            cv2.imshow(f"Sign: {sign} | press Q to skip to next", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms per frame
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    view_clips()