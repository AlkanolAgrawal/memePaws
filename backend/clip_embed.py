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

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return None

    # center crop (reduces background noise)
    crop = img_bgr[h//3:2*h//3, w//3:2*w//3]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)

    emb = emb.cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)
