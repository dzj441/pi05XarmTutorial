import zmq
import base64
import numpy as np
import pickle
import cv2


class VideoStreamer(object):
    def __init__(self, host, cam_port):
        self._init_socket(host, cam_port)

    def get_serial_num(self):
        return self.serial_num

    def _init_socket(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect("tcp://{}:{}".format(host, port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

    def get_image_tensor(self):
        raw_data = self.socket.recv()
        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)

        encoded_rgb = np.frombuffer(base64.b64decode(data["rgb_image"]), np.uint8)
        decoded_rgb = cv2.imdecode(encoded_rgb, cv2.IMREAD_COLOR)
        
        rgb_image = decoded_rgb
        # BGR to RGB
        # rgb_image = cv2.cvtColor(decoded_rgb, cv2.COLOR_BGR2RGB)

        return rgb_image