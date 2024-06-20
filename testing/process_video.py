"""
"""
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


def process_video(model, input_path, output_path, size=(2560, 1440)):
    """Process video from file"""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, 24.0, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, (size[0]//4, size[1]//4))

        start_time = time.time()

        frame_input = Image.fromarray(frame, "RGB")
        frame_input = np.expand_dims(frame_input, axis=0)
        sr1 = model.predict(frame[tf.newaxis, ...], verbose=0)
        sr1 = np.array(sr1, dtype=np.uint8)[0]

        print(f"Time taken step: {time.time()-start_time:.2f} sec\n")

        out.write(sr1)

        cv2.imshow("frame", sr1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
