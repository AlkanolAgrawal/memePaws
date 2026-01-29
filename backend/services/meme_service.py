import json
import cv2
import numpy as np
from functools import lru_cache
from services.embedding_service import similarity

DB_PATH = "assets/embeddings/meme_embeddings.json"


with open(DB_PATH, "r") as f:
    MEMES = json.load(f)


@lru_cache(maxsize=50)
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_best_meme(user_pose, user_emotion):
    user = {"pose": user_pose, "emotion": user_emotion}

    best = None
    best_score = -1

    for meme in MEMES:
        score = similarity(
            user,
            {
                "pose": np.array(meme["pose"]),
                "emotion": np.array(meme["emotion"])
            }
        )
        if score > best_score:
            best_score = score
            best = meme

    if best is None:
        return None

    return load_image(best["path"])
