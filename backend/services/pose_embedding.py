import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

_pose_landmarker = vision.PoseLandmarker.create_from_options(options)


def extract_pose_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = _pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]
    emb = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    return normalize(emb)


def normalize(vec):
    return (vec - vec.mean()) / (vec.std() + 1e-6)
