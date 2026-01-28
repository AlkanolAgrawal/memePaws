from PIL import Image, ImageDraw, ImageFont

def render_meme(image_path, caption):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # simple caption (top-left for now)
    draw.text((20, 20), caption, fill="white")

    return img
