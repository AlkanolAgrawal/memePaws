import json
import numpy as np
from collections import deque

with open("assets/embeddings.json", "r") as f:
    MEMES = json.load(f)

RECENT = deque(maxlen=12)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)

def softmax(x, temp=0.07):
    x = np.array(x)
    x = x / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def find_best(user_clip, emotion_vec):
    scored = []

    for m in MEMES:
        img_sim = cosine(user_clip, m["clip"])
        emo_sim = cosine(emotion_vec, m["clip"])

        score = 0.7 * img_sim + 0.3 * emo_sim

        if m["path"] in RECENT:
            score -= 0.4

        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)

    TOP_K = 8
    top = scored[:TOP_K]

    scores = [s for s, _ in top]
    probs = softmax(scores)

    idx = np.random.choice(len(top), p=probs)
    chosen = top[idx][1]

    RECENT.append(chosen["path"])
    return chosen, scores[idx]
