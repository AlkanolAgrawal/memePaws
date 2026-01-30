import json, cv2, numpy as np
from collections import deque

with open("assets/embeddings.json") as f:
    MEMES = json.load(f)

RECENT = deque(maxlen=5)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-6)

def get_best_meme(user_clip):
    best, best_score = None, -1

    for m in MEMES:
        score = cosine(user_clip, m["clip"])

        if m["path"] in RECENT:
            score -= 0.15   # diversity penalty

        if score > best_score:
            best_score, best = score, m

    if best_score < 0.25:
        return None

    RECENT.append(best["path"])
    img = cv2.imread(best["path"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
