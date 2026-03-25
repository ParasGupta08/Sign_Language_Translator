# 🤟 Sign Language Translator

A real-time sign language recognition system that captures hand gestures via webcam and converts them into spoken English using a CNN + GRU deep learning pipeline.

---

## 🧠 How It Works

The system uses a two-stage deep learning architecture:

1. **GestureCNN** — A convolutional neural network that extracts spatial features from each grayscale hand ROI frame (64×64px).
2. **TemporalGRU** — A GRU-based recurrent network that processes a sequence of CNN feature vectors to classify the gesture over time.

At inference, the webcam feed is processed in real-time. Skin-colour and motion detection isolate the hand ROI, a sliding window of frames is fed through the CNN+GRU, and majority voting over a batch of predictions triggers a spoken audio output via `pyttsx3`.

```
Webcam → Hand ROI (skin + motion) → CNN (per frame) → GRU (sequence) → Softmax → TTS
```

---

## 🗂️ Project Structure

```
Sign_Language_Translator/
│
├── model/
│   ├── cnn.py              # GestureCNN definition
│   └── temporal.py         # TemporalGRU definition
│
├── data/
│   ├── raw/                # Raw recorded video clips
│   └── processed/          # Preprocessed frame sequences (.npy)
│
├── checkpoints/
│   ├── best_model.pth      # Best checkpoint (saved on val accuracy improvement)
│   ├── last_model.pth      # Most recent checkpoint
│   └── training_curves.png # Loss/accuracy plots
│
├── Kaggle Trained Model/   # Pre-trained weights trained on Kaggle
│
├── config.py               # All hyperparameters and vocabulary settings
├── collect_data.py         # Webcam-based data collection script
├── preprocess.py           # Converts raw clips → frame sequences
├── dataset.py              # PyTorch Dataset + DataLoader
├── train.py                # Full CNN+GRU training pipeline
├── evaluate.py             # Model evaluation + confusion matrix
├── inference.py            # Real-time webcam inference + TTS
├── check_clips.py          # Utility to verify collected data
└── requirements.txt
```

---

## 🏷️ Supported Signs

The default vocabulary (configurable in `config.py`):

| Index | Sign |
|-------|------|
| 0 | HELLO |
| 1 | THANK YOU |
| 2 | YES |
| 3 | NO |
| 4 | PLEASE |

> To add or change signs, edit the `SIGNS` list in `config.py`. No other file needs to be changed.

---

## ⚙️ Installation

**Requirements:** Python 3.8+, a webcam, and optionally a CUDA-capable GPU.

```bash
# 1. Clone the repository
git clone https://github.com/ParasGupta08/Sign_Language_Translator.git
cd Sign_Language_Translator

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---------|---------|
| opencv-python | ≥ 4.8.0 |
| numpy | ≥ 1.24.0 |
| torch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| pyttsx3 | ≥ 2.90 |
| scikit-learn | ≥ 1.3.0 |
| matplotlib | ≥ 3.7.0 |
| seaborn | ≥ 0.12.0 |
| tqdm | ≥ 4.65.0 |
| Pillow | ≥ 10.0.0 |

> **Note for Linux users:** `pyttsx3` may require `espeak` — install with `sudo apt install espeak`.

---

## 🚀 Usage

### Step 1 — Collect Training Data

Run the data collection script and follow the on-screen prompts to record clips for each sign:

```bash
python collect_data.py
```

Each sign records **30 clips × 30 frames** by default. Clips are saved to `data/raw/`.

### Step 2 — Preprocess

Convert raw video clips into normalized frame sequences:

```bash
python preprocess.py
```

Output is saved to `data/processed/` as `.npy` files.

### Step 3 — Train

```bash
python train.py
```

Training runs for 60 epochs with AdamW + StepLR scheduling. The best model (by validation accuracy) is saved to `checkpoints/best_model.pth`. Training curves are saved as `checkpoints/training_curves.png`.

### Step 4 — Evaluate

```bash
python evaluate.py
```

Prints per-class accuracy and displays a confusion matrix.

### Step 5 — Run Real-Time Inference

```bash
python inference.py
```

Point your webcam at your hand and perform any of the trained signs. The system will:
- Draw a bounding box around the detected hand
- Display the predicted gesture and confidence on screen
- Speak the gesture name aloud when a stable, confident prediction is made

**Press `Q` to quit.**

---

## 🔧 Configuration

All settings are centralised in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIGNS` | `[HELLO, THANK YOU, YES, NO, PLEASE]` | Sign vocabulary |
| `CLIPS_PER_SIGN` | `30` | Training clips per sign |
| `FRAMES_PER_CLIP` | `30` | Frames per clip |
| `FRAME_SIZE` | `(64, 64)` | CNN input resolution |
| `CNN_FEATURE_DIM` | `128` | CNN output feature size |
| `GRU_HIDDEN_DIM` | `128` | GRU hidden state size |
| `EPOCHS` | `60` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Initial learning rate |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum softmax score to accept prediction |
| `SMOOTHING_WINDOW` | `4` | Majority-vote window size |
| `WEBCAM_INDEX` | `0` | Camera device index |

---

## 🗃️ Using the Pre-trained Model

A model trained on Kaggle is available in the `Kaggle Trained Model/` folder. To use it, copy the weights to `checkpoints/` and update `BEST_MODEL` in `config.py` if needed, then run:

```bash
python inference.py
```

---

## 📊 Model Architecture

**GestureCNN** (per-frame feature extractor)
- Input: `(1, 64, 64)` grayscale frame
- 3× Conv → BatchNorm → ReLU → MaxPool blocks
- Output: 128-dim feature vector

**TemporalGRU** (sequence classifier)
- Input: sequence of 30 CNN feature vectors `(T=30, F=128)`
- 2-layer GRU with hidden dim 128 and dropout 0.3
- Output: class logits `(NUM_CLASSES,)`

---

## 💡 Tips for Best Results

- Use the system in **good, consistent indoor lighting**
- Keep your hand within the **right half of the frame** where detection is active
- Avoid skin-coloured backgrounds (walls, clothing) in the shot
- Hold each gesture **steadily for ~1 second** to allow the smoothing window to stabilise
- Re-collect data if accuracy is low — the model is only as good as the training clips

---

## 🤝 Contributing

Pull requests are welcome. To add new signs, update the `SIGNS` list in `config.py`, re-collect data, and retrain.

---

## 📄 License

This project is open source. Feel free to use and modify it for educational and research purposes.