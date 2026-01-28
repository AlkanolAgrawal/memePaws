import os
import json
import random

# base directory where memes are stored
MEME_DIR = "assets/memes"

# load captions once (IMPORTANT)
with open("assets/captions.json", "r") as f:
    CAPTIONS = json.load(f)


def get_meme(trigger):
    """
    trigger: string like 'hands_on_head'
    returns:
        image_path (str)
        caption (str)
    """

    # folder corresponding to the trigger
    folder_path = os.path.join(MEME_DIR, trigger)

    # safety check
    if not os.path.exists(folder_path):
        return None, "No meme available"

    # list all images in folder
    images = [
        img for img in os.listdir(folder_path)
        if img.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        return None, "No meme available"

    # pick random meme image
    image_name = random.choice(images)
    image_path = os.path.join(folder_path, image_name)

    # pick random caption
    caption_list = CAPTIONS.get(trigger, [""])
    caption = random.choice(caption_list)

    return image_path, caption
