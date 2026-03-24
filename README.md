# MemePaws 🐾

**Emotion-Anchored Meme Engine**

MemePaws is an interactive webcam application that reads your facial expressions in real-time and overlays the perfect meme for the moment. Using OpenAI's CLIP model and advanced emotion detection, it matches your reactions with contextually relevant memes from a curated collection.

![MemePaws Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🎭 **Real-time Emotion Detection**: Analyzes your facial expressions using CLIP embeddings
- 🎯 **Smart Meme Matching**: Finds the most relevant meme based on your emotion and context
- 🔄 **Dynamic Overlay**: Seamlessly blends memes over your webcam feed
- 🎲 **Variety Control**: Uses softmax-based selection to avoid repetitive memes
- 🚀 **Easy to Use**: Simple Gradio interface for instant setup

## 🎭 Supported Emotions

The system recognizes and responds to 10 different emotional states:
- 😠 Angry
- 😕 Confused
- 😏 Sarcastic
- 😲 Shocked
- 😢 Sad
- 😭 Crying
- 😊 Happy
- 😂 Laughing
- 🤦 Facepalm
- 😐 Awkward Silence

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Webcam
- GPU recommended (CUDA) but CPU works too

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/memePaws.git
   cd memePaws
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your meme collection**
   - Add your meme images to `backend/assets/memes/`
   - Run the preprocessing script to generate embeddings:
     ```bash
     cd backend
     python preprocess_memes.py
     ```

4. **Launch the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   - Gradio will automatically open the interface (usually at `http://127.0.0.1:7860`)
   - Allow webcam access when prompted
   - Start making faces! 😄

## 🏗️ Project Structure

```
memePaws/
├── app.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── backend/
│   ├── main.py                 # Main Gradio application logic
│   ├── clip_embed.py           # CLIP embedding generation
│   ├── emotion_anchors.py      # Emotion text embeddings (CLIP-based)
│   ├── face_emotion.py         # Facial emotion detection (DeepFace)
│   ├── meme_matcher.py         # Core matching algorithm
│   ├── preprocess_memes.py     # Meme embedding preprocessor
│   └── assets/
│       ├── embeddings.json     # Pre-computed meme embeddings
│       └── memes/              # Your meme collection
└── frontend/
    ├── index.html
    ├── script.js
    └── style.css
```

## 🔧 How It Works

1. **Capture**: The webcam captures your face in real-time
2. **Embed**: CLIP generates a semantic embedding of your facial expression
3. **Analyze**: DeepFace detects your facial emotions and maps them to CLIP emotion anchors
4. **Match**: Memes are scored based on:
   - 70% visual similarity to your expression (CLIP embeddings)
   - 30% emotional alignment
   - -0.4 penalty for recently shown memes
5. **Select**: Softmax-based probabilistic selection ensures variety
6. **Display**: The best-matching meme is overlaid on your video feed with 80% opacity

## 🎨 Algorithm Details

### Similarity Thresholds
- **Entry Threshold**: 0.16 - Minimum similarity to display a meme
- **Exit Threshold**: 0.13 - Similarity below which the meme disappears
- **Minimum Display**: 15 frames - Prevents flickering

### Scoring Formula
```python
score = 0.7 × visual_similarity + 0.3 × emotional_similarity - repetition_penalty
```

### Anti-Repetition
- Maintains a buffer of the last 12 shown memes
- Applies a -0.4 penalty to recently displayed memes
- Uses temperature-scaled softmax for diverse selection

## 🛠️ Customization

### Add Your Own Memes
1. Add images (JPEG/PNG) to `backend/assets/memes/`
2. Run `python preprocess_memes.py` to regenerate embeddings
3. Restart the application

### Adjust Sensitivity
Edit thresholds in `backend/main.py`:
```python
SIM_ENTER = 0.16      # Lower = more sensitive
SIM_EXIT = 0.13       # Lower = memes stay longer
MIN_ONSCREEN = 15     # Higher = more stable display
```

Edit emotion detection threshold:
```python
if dominant_emotion == 'neutral' or emotion_strength < 0.3:
    # Increase 0.3 for less sensitive, decrease for more sensitive
```

### Modify Emotion Categories
Edit the emotion list in `emotion_anchors.py`:
```python
EMOTIONS = [
    "your custom emotion",
    # ... more emotions
]
```

## 📦 Dependencies

- **PyTorch & Torchvision**: Deep learning framework
- **OpenAI CLIP**: Vision-language model for semantic embeddings
- **DeepFace**: Facial emotion recognition
- **TensorFlow & tf-keras**: Required by DeepFace
- **OpenCV**: Computer vision and image processing
- **Gradio**: Web interface
- **NumPy**: Numerical computing
- **Pillow**: Image processing

See [requirements.txt](requirements.txt) for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Add more emotion categories
- Switch between DeepFace and CLIP-based emotion detection
- Create a meme rating/feedback system
- Add sound effects
- Build a mobile version
- Optimize for real-time performance
- Add multi-face support

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the powerful vision-language model
- [Gradio](https://gradio.app/) for the fantastic interface framework
- The internet for the endless supply of memes 🎉

## 📧 Contact

For questions or suggestions, feel free to open an issue or reach out!

---

**Made with ❤️ and memes**
