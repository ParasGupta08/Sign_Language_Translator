import cv2
import numpy as np
import torch
import collections
import time
import threading
import queue
import os
from gtts import gTTS
import pygame

from config import (
    FRAME_SIZE, SEQUENCE_LEN, CNN_FEATURE_DIM,
    SKIN_HSV_LOWER, SKIN_HSV_UPPER, ROI_PADDING,
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW,
    BEST_MODEL, WEBCAM_INDEX, IDX_TO_SIGN
)
from model.cnn      import GestureCNN
from model.temporal import TemporalGRU

# ─────────────────────────────────────────
# gTTS Background Worker (Non-Blocking)
# ─────────────────────────────────────────

speech_queue = queue.Queue()

def tts_worker():
    """Background thread to handle Google TTS without freezing the webcam."""
    pygame.mixer.init()
    while True:
        text = speech_queue.get()
        if text is None: break
        
        # Debounce: Skip if a newer sign is already waiting in the queue
        if speech_queue.qsize() > 1:
            speech_queue.task_done()
            continue

        try:
            # Generate speech
            tts = gTTS(text=text, lang='en')
            temp_file = f"speech_{int(time.time())}.mp3"
            tts.save(temp_file)
            
            # Play speech
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.unload()
            # Clean up file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
            
        speech_queue.task_done()

# Start TTS thread
threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    """Adds text to the queue and clears any old pending speech."""
    with speech_queue.mutex:
        speech_queue.queue.clear() 
    speech_queue.put(text)

# ─────────────────────────────────────────
# Hand ROI Extraction
# ─────────────────────────────────────────

def extract_hand_roi(frame):
    fh, fw = frame.shape[:2]
    zone_x_start = int(fw * 0.35)
    zone = frame[:, zone_x_start:]

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 15, 60]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 1500: return None, None

    x, y, w, h = cv2.boundingRect(best)
    x += zone_x_start
    cx, cy = x + w // 2, y + h // 2
    side = max(w, h, int(min(fw, fh) * 0.25)) + ROI_PADDING
    
    x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
    x2, y2 = min(fw, cx + side//2), min(fh, cy + side//2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None, None
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, FRAME_SIZE)
    return roi_gray, (x1, y1, x2 - x1, y2 - y1)

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────

def load_models(device):
    cnn = GestureCNN().to(device)
    gru = TemporalGRU().to(device)
    ckpt = torch.load(BEST_MODEL, map_location=device)
    cnn.load_state_dict(ckpt["cnn_state"])
    gru.load_state_dict(ckpt["gru_state"])
    cnn.eval(); gru.eval()
    print(f"✓ Models loaded on {device}")
    return cnn, gru

@torch.no_grad()
def predict(cnn, gru, frame_buffer, device):
    clip = np.stack(frame_buffer, axis=0)
    clip = clip[:, np.newaxis, :, :]
    clip_t = torch.from_numpy(clip).float().unsqueeze(0).to(device)
    B, T, C, H, W = clip_t.shape
    features = cnn(clip_t.view(B*T, C, H, W)).view(B, T, CNN_FEATURE_DIM)
    logits = gru(features)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(dim=0)
    return idx.item(), conf.item()

# ─────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn, gru = load_models(device)
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    frame_buffer = []
    pred_history = collections.deque(maxlen=10) # Smoothing window
    
    # State tracking
    last_spoken_sign = None
    curr_sign = "Scanning..."
    curr_conf = 0.0
    stable_counter = 0

    print("\n[INFO] AI is active. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        roi_gray, bbox = extract_hand_roi(frame)

        if roi_gray is not None:
            norm = roi_gray.astype(np.float32) / 255.0
            frame_buffer.append(norm)
            if len(frame_buffer) > SEQUENCE_LEN: 
                frame_buffer.pop(0)
        else:
            # Clear state when hand is gone
            frame_buffer.clear()
            pred_history.clear()
            last_spoken_sign = None
            stable_counter = 0
            curr_sign = "No Hand"
            curr_conf = 0.0

        if len(frame_buffer) == SEQUENCE_LEN:
            idx, conf = predict(cnn, gru, frame_buffer, device)
            pred_history.append(idx)

            if len(pred_history) >= 5:
                vote, count = collections.Counter(pred_history).most_common(1)[0]
                sign_name = IDX_TO_SIGN[vote]
                
                # High confidence check (0.80 recommended)
                if conf > 0.80 and count >= (len(pred_history) // 2 + 1):
                    curr_sign = sign_name
                    curr_conf = conf

                    # Logic: Speak only if sign is different from last spoken
                    if sign_name != last_spoken_sign:
                        stable_counter += 1
                        # Wait for 5 stable frames of the NEW sign
                        if stable_counter >= 5:
                            print(f"  → Result: {sign_name}")
                            speak(sign_name)
                            last_spoken_sign = sign_name
                            stable_counter = 0
                else:
                    curr_sign = "..."
                    # We don't reset last_spoken_sign here to maintain memory
                    # during minor flickers.

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
        
        cv2.putText(frame, f"Sign: {curr_sign}", (20, 50), 2, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {curr_conf:.2f}", (20, 90), 2, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Sign Language AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()