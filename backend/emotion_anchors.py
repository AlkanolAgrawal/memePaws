import clip
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = [
    "angry reaction",
    "confused reaction",
    "sarcastic reaction",
    "shocked reaction",
    "sad reaction",
    "crying reaction",
    "happy reaction",
    "laughing reaction",
    "facepalm reaction",
    "awkward silence"
]

model, _ = clip.load("ViT-B/32", device=device)
tokens = clip.tokenize(EMOTIONS).to(device)

with torch.no_grad():
    TEXT_EMB = model.encode_text(tokens)
    TEXT_EMB /= TEXT_EMB.norm(dim=-1, keepdim=True)

TEXT_EMB = TEXT_EMB.cpu().numpy()