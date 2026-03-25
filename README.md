# 🤟 Hand Sign Recognition System

A real-time hand gesture recognition app built with **Streamlit**, **MediaPipe**, and **scikit-learn**.

---

## 📁 Project Structure

```
hand_sign_recognition/
├── app.py              # Main Streamlit application
├── model_utils.py      # Feature extraction + classifier
├── train.py            # Standalone training script
├── requirements.txt    # Python dependencies
├── README.md
├── dataset/            # ← YOUR DATASET GOES HERE
│   ├── A/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── B/
│   │   └── ...
│   └── Z/
│       └── ...
├── hand_sign_model.pkl    # Generated after training
└── hand_sign_labels.pkl   # Generated after training
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

Place your dataset inside the `dataset/` folder.  
Each sub-folder name = the letter label (A–Z or custom).

```
dataset/
├── A/   → images of the "A" hand sign
├── B/   → images of the "B" hand sign
...
```

### 3. Train the model

**Option A – via terminal:**
```bash
python train.py --dataset ./dataset
```

**Option B – via the app:**  
Open the app → sidebar → **Dataset & Training** → click "Train/Retrain Model".

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📷 **Live Webcam** | Dual-view: raw feed + annotated recognition side by side |
| 📁 **Video Upload** | Process any MP4/AVI/MOV file frame by frame |
| 🔤 **Letter Display** | Large letter overlay with confidence percentage |
| 🧠 **Custom Dataset** | Works with any folder-based image dataset |
| ⚡ **Fast Inference** | MediaPipe landmarks + Random Forest = ~10ms/frame |

---

## 🧠 How it works

1. **MediaPipe Hands** detects 21 3D keypoints on the hand
2. Keypoints are normalised (translation + scale invariant)
3. A **Random Forest** classifier maps the 63-dim feature vector to a letter
4. Confidence score is shown alongside the prediction

---

## 📝 Dataset Tips

- **100+ images per letter** for good accuracy  
- Vary backgrounds, lighting, and hand sizes  
- Consistent framing (hand centred, clear background) improves results  
- Popular public dataset: [ASL Alphabet on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "No trained model found" | Run `python train.py` first |
| Low accuracy | Add more diverse training images |
| Webcam not opening | Check camera permissions / index (default 0) |
| Slow processing | Increase "skip frames" slider in video mode |
