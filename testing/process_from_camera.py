"""
This script is used to process the input from the camera using the model.
"""

import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


def process_from_camera(model, video_capture=0, size=(800, 600)):
    """
    Process the input from the camera using the model.
    """
    cap = cv2.VideoCapture(video_capture)

    while True:
        _, frame_original = cap.read()
        frame_original = cv2.resize(frame_original, size)
        frame = cv2.resize(
            frame_original, (size[0] // 4, size[1] // 4), interpolation=cv2.INTER_AREA
        )

        frame_input = Image.fromarray(frame, "RGB")
        frame_input = np.expand_dims(frame_input, axis=0)
        start = time.time()
        sr1 = model.predict(frame[tf.newaxis, ...], verbose=0)
        sr1 = np.array(sr1, dtype=np.uint8)

        print(f"Time taken step: {time.time()-start:.2f} sec\n")

        cv2.imshow("output", sr1[0])
        cv2.imshow("frame", frame)
        cv2.imshow("original", frame_original)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
