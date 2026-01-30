import cv2, gradio as gr
from clip_embed import clip_embed_bgr
from meme_matcher import get_best_meme

def run(frame):
    if frame is None:
        return None

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    user_clip = clip_embed_bgr(bgr)

    meme = get_best_meme(user_clip)
    if meme is None:
        return frame

    meme = cv2.resize(meme, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(meme, 0.8, frame, 0.2, 0)

gr.Interface(
    run,
    gr.Image(sources=["webcam"], streaming=True),
    gr.Image()
).launch()
