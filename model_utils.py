import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path

MODEL_PATH = "hand_sign_model.pkl"
LABEL_PATH = "hand_sign_labels.pkl"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_landmarks(rgb_image, hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = lm[0]

    coords = []
    for p in lm:
        coords.extend([p.x - wrist.x, p.y - wrist.y, p.z - wrist.z])

    arr = np.array(coords, dtype=np.float32)

    xs, ys = arr[0::3], arr[1::3]
    scale = np.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2) + 1e-6
    return arr / scale


class HandSignClassifier:
    def __init__(self):
        self.model = None
        self.classes = []
        self.load()

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
            try:
                self.model = pickle.load(open(MODEL_PATH, "rb"))
                self.classes = pickle.load(open(LABEL_PATH, "rb"))
            except:
                self.model, self.classes = None, []

    def save(self):
        pickle.dump(self.model, open(MODEL_PATH, "wb"))
        pickle.dump(self.classes, open(LABEL_PATH, "wb"))

    def is_trained(self):
        return self.model is not None and len(self.classes) > 0

    def get_classes(self):
        return self.classes

    def train(self, dataset_path):
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print("Dataset not found")
            return False

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

        X, y = [], []
        labels = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

        for label in labels:
            for img_path in (dataset_path / label).iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    feat = extract_landmarks(rgb, results.multi_hand_landmarks[0])
                    X.append(feat)
                    y.append(label)

        hands.close()

        if not X:
            print("No data found")
            return False

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        self.model = model
        self.classes = labels
        self.save()

        print("Model trained successfully")
        return True

    def predict(self, rgb_image, hand_landmarks):
        if not self.is_trained():
            return None, 0.0

        try:
            feat = extract_landmarks(rgb_image, hand_landmarks).reshape(1, -1)
            probs = self.model.predict_proba(feat)[0]
            idx = np.argmax(probs)
            return self.model.classes_[idx], float(probs[idx])
        except:
            return None, 0.0