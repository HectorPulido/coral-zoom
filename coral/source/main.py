import socket
import pickle

import time
import numpy as np
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

HOST = "0.0.0.0"
PORT = 5069
MODEL = "model_edgetpu.tflite"


def generate_model():
    interpreter = make_interpreter(MODEL)
    interpreter.allocate_tensors()
    return interpreter


def inference_model(interpreter, data_img):
    common.set_input(interpreter, data_img)
    interpreter.invoke()
    return common.output_tensor(interpreter, 0).copy()


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


def socket_server():
    print("Starting server")

    interpreter = generate_model()
    server_socket = socket.socket()
    server_socket.bind((HOST, PORT))

    server_socket.listen(1)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))
    while True:
        data = recieve_big_data(conn)
        if not data:
            break

        data_img = pickle.loads(data)

        classes = inference_model(interpreter, data_img)

        # Enviar el resultado de vuelta al cliente
        results_pickle = pickle.dumps(classes)
        results_pickle_len = len(results_pickle).to_bytes(32, byteorder="big")
        conn.sendall(results_pickle_len)
        conn.sendall(results_pickle)

    conn.close()


if __name__ == "__main__":
    socket_server()
