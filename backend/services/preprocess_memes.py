import os
import json
import cv2
import tempfile

from services.pose_embedding import extract_pose_embedding
from services.emotion_embedding import extract_emotion_embedding

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

    if not db:
        print("[WARN] No valid memes found. Database not updated.")
        return

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        json.dump(db, tmp, indent=2)
        temp_name = tmp.name

    os.replace(temp_name, OUT_PATH)
    print(f"[OK] Saved {len(db)} meme embeddings")
