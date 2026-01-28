import cv2
import math
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

pose_landmarker = vision.PoseLandmarker.create_from_options(options)


def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def detect_gesture(frame):
    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]

    head = lm[0]
    lw = lm[15]
    rw = lm[16]
    left_shoulder = lm[11]
    right_shoulder = lm[12]

    # scale based on body width
    head_radius = distance(left_shoulder, right_shoulder) * 0.8

    # hands close to head region
    if (
        distance(lw, head) < head_radius and
        distance(rw, head) < head_radius and
        lw.y < left_shoulder.y and
        rw.y < right_shoulder.y
    ):
        return "hands_on_head"
    return None
