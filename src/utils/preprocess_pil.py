from PIL import Image
import numpy as np
from utils.center_image import center_image
from config.config import Config


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    # Ensure white background (handles RGBA from canvas)
    if pil_img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[-1])
        pil_img = bg.convert("L")
    else:
        pil_img = pil_img.convert("L")

    # Resize to 28x28
    pil_img = pil_img.resize((28, 28), Image.BILINEAR)

    x = np.array(pil_img).astype("float32")  # 0..255

    if Config.INVERT:
        x = 255.0 - x

    # normalize like training
    if Config.NORM == "0_1":
        x = x / 255.0

    # optional light binarize helps thin strokes
    # x = (x > 0.2).astype("float32")

    # center like many MNIST demos (optional but helps)
    x = center_image(x)

    # shape for your MLP
    x = x.reshape(1, 28 * 28)  # (1, 784)
    return x
