import torch
import clip
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_embed_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = preprocess(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img)

    emb = emb.cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)
