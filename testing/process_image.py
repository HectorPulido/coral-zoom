"""
This script processes an image using the model and saves the output.
"""

import numpy as np
from PIL import Image


def process_image(model, image_input, image_output):
    img = Image.open(image_input)
    img = np.array(img)

    # Remove the alpha channel if it exists
    if img.shape[-1] == 4:
        img = img[:, :, 0:3]

    frame_input = Image.fromarray(img, "RGB")
    frame_input = np.expand_dims(frame_input, axis=0)
    sr1 = model.predict(frame_input)

    sr1 = np.array(sr1, dtype=np.uint8)
    sr1 = Image.fromarray(sr1[0], "RGB")
    sr1.save(image_output)
