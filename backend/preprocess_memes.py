import os, json, cv2
from clip_embed import clip_embed_bgr

MEME_DIR = "assets/memes"
OUT = "assets/embeddings.json"

db = []

for f in os.listdir(MEME_DIR):
    path = os.path.join(MEME_DIR, f)
    if not f.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    db.append({
        "path": path,
        "clip": clip_embed_bgr(img).tolist()
    })

    print("Indexed:", f)

with open(OUT, "w") as f:
    json.dump(db, f)

print("✅ Total memes indexed:", len(db))
