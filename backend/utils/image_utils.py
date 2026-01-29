import cv2

def overlay(frame, meme, alpha=0.8):
    h, w = frame.shape[:2]
    meme = cv2.resize(meme, (w, h))
    return cv2.addWeighted(meme, alpha, frame, 1 - alpha, 0)
