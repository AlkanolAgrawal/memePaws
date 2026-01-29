import gradio as gr
import cv2
import numpy as np

from services.pose_embedding import extract_pose_embedding
from services.emotion_embeddings import extract_emotion_embedding
from services.meme_service import get_best_meme
from utils.image_utils import overlay
from services.preprocess_memes import preprocess

preprocess()  # safe to call; cached by JSON


def run(frame):
    if frame is None:
        return None

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    pose = extract_pose_embedding(bgr)
    if pose is None:
        return frame

    emotion = extract_emotion_embedding(bgr)
    meme = get_best_meme(pose, emotion)

    if meme is None:
        return frame

    return overlay(frame, meme)


with gr.Blocks() as demo:
    gr.Markdown("# MemePaws – Pose + Emotion Meme Matching")

    cam = gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Webcam"
    )

    out = gr.Image(label="Output")

    cam.stream(run, inputs=cam, outputs=out)

demo.launch()
