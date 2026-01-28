import cv2
import gradio as gr

from pose_service import detect_gesture   # adjust import if in services/

def run(image):
    """
    image: numpy array in RGB (given by Gradio)
    """

    # Gradio gives RGB, OpenCV/MediaPipe expect BGR
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect gesture
    gesture = detect_gesture(frame)

    # Return result for verification
    if gesture is None:
        return "No gesture detected"
    return f"Detected gesture: {gesture}"


gr.Interface(
    fn=run,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="MemePaws – Gesture Detection Test",
    description="Upload an image and test hands-on-head gesture detection"
).launch()
