import cv2
import mediapipe as mp
import math 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    
    lm = result.pose_landmarks.landmark
    head = lm[0]
    lw = lm[15]
    rw = lm[16]

    if lw.y < head.y and rw.y < head.y:
        if distance(lw, head) < 0.25 and distance(rw, head) < 0.25:
            return "hands_on_head"

    return None