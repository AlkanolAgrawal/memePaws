import json
import numpy as np
import cv2
from collections import deque

EMBED_PATH = "assets/embeddings/meme_embeddings.json"

with open(EMBED_PATH, "r") as f:
    MEMES = json.load(f)

RECENT = deque(maxlen=5)
THRESHOLD = 0.8   # confidence gate


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


def get_best_meme(pose_vec, emo_vec):
    best = None
    best_score = float("inf")

    for m in MEMES:
        score = 0.0

        if pose_vec is not None:
            score += 0.7 * cosine(pose_vec, m["pose"])

        if emo_vec is not None:
            score += 0.3 * cosine(emo_vec, m["emotion"])

        # diversity penalty
        if m["path"] in RECENT:
            score += 0.15

        if score < best_score:
            best_score = score
            best = m

    if best is None or best_score > THRESHOLD:
        return None

    RECENT.append(best["path"])
    img = cv2.imread(best["path"])
    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
