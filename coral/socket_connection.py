import time
import pickle
import socket
from PIL import Image
import numpy as np


def recieve_big_data(conn, len_size=32, data_size=4096):
    data_len = conn.recv(len_size)
    if not data_len:
        return None

    data_len = int.from_bytes(data_len, byteorder="big")

    data = b""
    while len(data) < data_len:
        packet = conn.recv(data_size)
        if not packet:
            return data
        data += packet

    return data


HOST = "0.0.0.0"
PORT = 5069

client_socket = socket.socket()  # instantiate
client_socket.connect((HOST, PORT))  # connect to the server

for i in range(100):
    # TODO: change this for the screen capture
    image = (
        Image.open("test_image.jpg").convert("RGB").resize((512, 512), Image.LANCZOS)
    )
    pickle_img = pickle.dumps(image)

    start_time = time.time()
    # Send data
    client_socket.send(len(pickle_img).to_bytes(32, byteorder="big"))
    client_socket.send(pickle_img)

    # Recieve data
    data = recieve_big_data(client_socket)
    data = pickle.loads(data)
    print("%.1fms" % ((time.time() - start_time) * 1000))

sr1 = np.array(data, dtype=np.uint8)
sr1 = Image.fromarray(sr1[0], "RGB")
sr1.save("test_image_output.jpg")
