import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

_face_landmarker = vision.FaceLandmarker.create_from_options(options)


def extract_emotion_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = _face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return np.zeros(32, dtype=np.float32)

    lm = result.face_landmarks[0]

    # select expressive regions (mouth + eyebrows)
    idxs = (
        list(range(61, 68)) +    # mouth
        list(range(105, 113)) +  # left eyebrow
        list(range(334, 342))    # right eyebrow
    )

    emb = []
    for i in idxs:
        emb.extend([lm[i].x, lm[i].y])

    emb = np.array(emb, dtype=np.float32)
    return (emb - emb.mean()) / (emb.std() + 1e-6)
