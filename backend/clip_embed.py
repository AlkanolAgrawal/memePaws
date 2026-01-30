import torch
import clip
import cv2
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_embed_bgr(img_bgr):
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)   # ✅ critical

    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)

    emb = emb.cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)
