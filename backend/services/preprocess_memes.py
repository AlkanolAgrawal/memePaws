import os
import json
import cv2
from services.pose_embedding import extract_pose_embedding
from services.emotion_embeddings import extract_emotion_embedding

MEME_DIRS = [
    "assets/memes_base",
    "assets/memes_user"
]

OUT_PATH = "assets/embeddings/meme_embeddings.json"


def preprocess():
    db = []

    for base in MEME_DIRS:
        if not os.path.exists(base):
            continue

        for root, _, files in os.walk(base):
            for f in files:
                if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                path = os.path.join(root, f)
                img = cv2.imread(path)
                if img is None:
                    continue

                pose = extract_pose_embedding(img)
                if pose is None:
                    continue

                emotion = extract_emotion_embedding(img)

                db.append({
                    "path": path,
                    "pose": pose.tolist(),
                    "emotion": emotion.tolist()
                })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(db, f)
