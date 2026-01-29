import numpy as np

POSE_WEIGHT = 0.75
EMOTION_WEIGHT = 0.25


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


def similarity(user, meme):
    pose_sim = cosine(user["pose"], meme["pose"])
    emo_sim = cosine(user["emotion"], meme["emotion"])

    return POSE_WEIGHT * pose_sim + EMOTION_WEIGHT * emo_sim
