import cv2
import numpy as np
from deepface import DeepFace

# Emotion mapping for consistency
EMOTION_MAP = {
    'angry': 'angry',
    'disgust': 'disgusted',
    'fear': 'scared',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

def detect_emotion(img_bgr):
    """
    Detect dominant emotion from a face image
    
    Args:
        img_bgr: OpenCV BGR image
        
    Returns:
        tuple: (emotion_name, confidence_score)
        Returns (None, 0.0) if no emotion detected
    """
    if img_bgr is None:
        return None, 0.0
    
    try:
        # DeepFace expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Analyze emotions
        result = DeepFace.analyze(
            img_rgb, 
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Handle both single face and multiple faces
        if isinstance(result, list):
            result = result[0]
        
        # Get emotion probabilities
        emotions = result['emotion']
        
        # Find dominant emotion
        dominant_emotion = result['dominant_emotion']
        confidence = emotions[dominant_emotion] / 100.0  # Convert to 0-1 scale
        
        # Map to our emotion labels
        mapped_emotion = EMOTION_MAP.get(dominant_emotion, dominant_emotion)
        
        return mapped_emotion, confidence
        
    except Exception as e:
        # If detection fails, return None
        return None, 0.0


def get_emotion_vector(img_bgr):
    """
    Get full emotion probability vector
    
    Args:
        img_bgr: OpenCV BGR image
        
    Returns:
        dict: Emotion probabilities {'angry': 0.1, 'happy': 0.7, ...}
    """
    if img_bgr is None:
        return {}
    
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        result = DeepFace.analyze(
            img_rgb, 
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        
        # Convert to 0-1 scale and map labels
        mapped_emotions = {
            EMOTION_MAP.get(k, k): v / 100.0 
            for k, v in emotions.items()
        }
        
        return mapped_emotions
        
    except Exception:
        return {}
