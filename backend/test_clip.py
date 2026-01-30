import os
import cv2
from clip_embed import clip_embed_bgr

MEME_DIR = "assets/memes"

files = os.listdir(MEME_DIR)
print("Files found:", files)

img_path = os.path.join(MEME_DIR, files[0])
img = cv2.imread(img_path)

print("Image loaded:", img is not None)

emb = clip_embed_bgr(img)

print("Embedding type:", type(emb))
print("Embedding shape:", emb.shape)
