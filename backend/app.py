import gradio as gr
import cv2
import numpy as np
import os

from services.pose_detection import detect_gesture


MEME_PATH = os.path.join(
    "assets", "memes", "hands_on_head", "panic", "CGG.png"
)


def run(image):
    # PIL → OpenCV (BGR)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gesture = detect_gesture(frame)

    if gesture == "hands_on_head":
        meme = cv2.imread(MEME_PATH)

        # safety check
        if meme is not None:
            return cv2.cvtColor(meme, cv2.COLOR_BGR2RGB)

    # always return an image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


with gr.Blocks() as demo:
    gr.Markdown("# MemePaws – Gesture to Meme")
    gr.Markdown("Upload an image with hands on head to get a meme")

    inp = gr.Image(type="pil", label="image")
    out = gr.Image(label="output")

    gr.Button("Submit").click(run, inp, out)
    gr.Button("Clear").click(lambda: None, None, out)

demo.launch()
