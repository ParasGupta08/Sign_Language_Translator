# =============================================================================
# config.py — Central configuration for the entire project
# To change your sign vocabulary, just update the SIGNS list below.
# =============================================================================

# ──────────────────────────────────────────────
# SIGN VOCABULARY  ← update these 10 when ready
# ──────────────────────────────────────────────

#  "STOP",
    #"WATER",
   # "HELP",
  #  "SORRY",
 #  "COME" ,
 
SIGNS = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "PLEASE",  
]

NUM_CLASSES = len(SIGNS)
LABEL_MAP   = {sign: idx for idx, sign in enumerate(SIGNS)}
IDX_TO_SIGN = {idx: sign for sign, idx in LABEL_MAP.items()}

# ──────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────
CLIPS_PER_SIGN      = 30        # number of video clips recorded per sign
FRAMES_PER_CLIP     = 30        # frames sampled from each clip
FRAME_SIZE          = (64, 64)  # (width, height) fed into CNN
DATA_DIR            = "data/raw"
PROCESSED_DIR       = "data/processed"

# ──────────────────────────────────────────────
# HAND / ROI ISOLATION
# ──────────────────────────────────────────────
# HSV skin-colour range (works under typical indoor lighting)
SKIN_HSV_LOWER = (0,  20,  70)
SKIN_HSV_UPPER = (20, 255, 255)
ROI_PADDING    = 20             # pixels of padding around detected hand contour

# ──────────────────────────────────────────────
# CNN
# ──────────────────────────────────────────────
CNN_IN_CHANNELS  = 1            # grayscale ROI frames
CNN_BASE_FILTERS = 32           # first conv layer filter count
CNN_FEATURE_DIM  = 128          # flattened feature vector size out of CNN

# ──────────────────────────────────────────────
# TEMPORAL MODEL (GRU)
# ──────────────────────────────────────────────
SEQUENCE_LEN   = FRAMES_PER_CLIP   # sliding window length (frames)
GRU_HIDDEN_DIM = 128
GRU_LAYERS     = 2
DROPOUT        = 0.3

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS         = 60
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
LR_STEP_SIZE   = 20             # StepLR scheduler step
LR_GAMMA       = 0.5
TRAIN_SPLIT    = 0.8
MODEL_SAVE_DIR = "checkpoints"
BEST_MODEL     = "checkpoints/best_model.pth"

# ──────────────────────────────────────────────
# REAL-TIME INFERENCE
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.5   # minimum softmax score to accept a prediction
SMOOTHING_WINDOW      = 4    # majority-vote over last N predictions
COOLDOWN_FRAMES       = 0   # frames to wait before repeating same sign (suppress duplicates)
WEBCAM_INDEX          = 0

# ──────────────────────────────────────────────
# AUGMENTATION
# ──────────────────────────────────────────────
AUG_HFLIP_PROB        = 0.5
AUG_BRIGHTNESS_RANGE  = (0.7, 1.3)
AUG_ROTATION_DEGREES  = 15
AUG_NOISE_STD         = 0.02
