import cv2
import gradio as gr
import numpy as np
from clip_embed import clip_embed_bgr
from meme_matcher import find_best
from emotion_anchors import TEXT_EMB, EMOTIONS

ACTIVE_MEME = None
ONSCREEN = 0

SIM_ENTER = 0.16
SIM_EXIT = 0.13
MIN_ONSCREEN = 15

def run(frame):
    global ACTIVE_MEME, ONSCREEN

    if frame is None:
        return None

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    user_clip = clip_embed_bgr(bgr)
    if user_clip is None:
        return frame

    # ---- emotion detection ----
    emotion_scores = TEXT_EMB @ user_clip
    idx = np.argmax(emotion_scores)
    emotion_strength = emotion_scores[idx]
    emotion_vec = TEXT_EMB[idx]

    # ignore neutral frames
    if emotion_strength < 0.25:
        ACTIVE_MEME = None
        ONSCREEN = 0
        return frame

    meme, score = find_best(user_clip, emotion_vec)

    if ACTIVE_MEME is None:
        if score >= SIM_ENTER:
            ACTIVE_MEME = meme
            ONSCREEN = 0
    else:
        ONSCREEN += 1
        if ONSCREEN >= MIN_ONSCREEN:
            if score <= SIM_EXIT:
                ACTIVE_MEME = None
                ONSCREEN = 0
            elif meme["path"] != ACTIVE_MEME["path"]:
                ACTIVE_MEME = meme
                ONSCREEN = 0

    if ACTIVE_MEME is None:
        return frame

    meme_img = cv2.imread(ACTIVE_MEME["path"])
    meme_img = cv2.cvtColor(meme_img, cv2.COLOR_BGR2RGB)
    meme_img = cv2.resize(meme_img, (frame.shape[1], frame.shape[0]))

    return cv2.addWeighted(meme_img, 0.8, frame, 0.2, 0)

with gr.Blocks() as demo:
    gr.Markdown("## MemePaws – Emotion Anchored Meme Engine 🐾")

    cam = gr.Image(sources=["webcam"], streaming=True)
    out = gr.Image()

    cam.stream(run, inputs=cam, outputs=out)

demo.launch()
