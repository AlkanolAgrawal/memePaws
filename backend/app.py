import gradio as gr
import cv2
import numpy as np

FRAME_SKIP = 3
frame_count = 0

POSE_BUF = []
EMO_BUF = []
BUF_SIZE = 5

from services.pose_embedding import extract_pose_embedding
from services.emotion_embedding import extract_emotion_embedding
from services.meme_service import get_best_meme


def run(frame):
    global frame_count, POSE_BUF, EMO_BUF

    if frame is None:
        return None

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        return frame

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (640, 480))

    pose_vec = extract_pose_embedding(bgr)
    emo_vec = extract_emotion_embedding(bgr)

    if pose_vec is None and emo_vec is None:
        return frame

    if pose_vec is not None:
        POSE_BUF.append(pose_vec)
        POSE_BUF = POSE_BUF[-BUF_SIZE:]
        pose_vec = np.mean(POSE_BUF, axis=0)

    if emo_vec is not None:
        EMO_BUF.append(emo_vec)
        EMO_BUF = EMO_BUF[-BUF_SIZE:]
        emo_vec = np.mean(EMO_BUF, axis=0)

    meme = get_best_meme(pose_vec, emo_vec)
    if meme is None:
        return frame

    meme = cv2.resize(meme, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(meme, 0.8, frame, 0.2, 0)


with gr.Blocks() as demo:
    gr.Markdown("# 🐾 MemePaws – Real-Time Meme Matching")

    cam = gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Webcam"
    )

    out = gr.Image(label="Output")

    cam.stream(run, inputs=cam, outputs=out)

demo.launch()
