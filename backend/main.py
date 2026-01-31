import cv2
import gradio as gr
import numpy as np
from clip_embed import clip_embed_bgr
from meme_matcher import find_best
from emotion_anchors import TEXT_EMB, EMOTIONS
from face_emotion import get_emotion_vector

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

    # ---- emotion detection with DeepFace ----
    emotion_dict = get_emotion_vector(bgr)
    if not emotion_dict:
        return frame
    
    # Get dominant emotion
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    emotion_strength = emotion_dict[dominant_emotion]
    
    # ignore neutral/weak frames
    if dominant_emotion == 'neutral' or emotion_strength < 0.3:
        ACTIVE_MEME = None
        ONSCREEN = 0
        return frame
    
    # Create emotion vector by matching to CLIP emotion embeddings
    # Map DeepFace emotions to our CLIP emotion anchors
    emotion_mapping = {
        'angry': 'angry reaction',
        'disgusted': 'facepalm reaction',
        'scared': 'shocked reaction',
        'happy': 'happy reaction',
        'sad': 'sad reaction',
        'surprised': 'shocked reaction',
        'neutral': 'awkward silence'
    }
    
    emotion_text = emotion_mapping.get(dominant_emotion, 'happy reaction')
    emotion_idx = EMOTIONS.index(emotion_text) if emotion_text in EMOTIONS else 0
    emotion_vec = TEXT_EMB[emotion_idx]

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

