import base64
import json
import os
import sys
import time

import keras.applications.mobilenet as mobilenet
import numpy as np
import scipy.misc
import skvideo.io
from keras.models import load_model

import cv2

class_bg = 0
class_car = 1
class_road = 2
n_classes = 3

image_shape = (600, 800)
out_shape = (416, 800)
out_shape = [x // 32 * 32 for x in out_shape]

MODEL_DIR = "model/"
model_path = os.path.join(MODEL_DIR, 'epoch-007-val_loss-0.0428.hdf5')


def preprocess_input(x):
    x /= 127.5
    x -= 1.0
    return x.astype(np.float32)


if __name__ == '__main__':
    file = sys.argv[-1]

    if file == 'demo.py':
        print("Error loading video")
        quit

    # Define encoder function
    def encode(array):
        retval, buff = cv2.imencode('.png', array)
        return base64.b64encode(buff).decode("utf-8")

    video = skvideo.io.vread(file)

    m = load_model(
        model_path,
        custom_objects={
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D
        })

    answer_key = {}
    # Frame numbering starts at 1
    frame = 1
    
    pr = np.zeros([*image_shape])
    for rgb_frame in video:

        # rgb_frame = preprocess_input(rgb_frame)

        pr_out = m.predict(np.array([rgb_frame[-out_shape[0]:, :, :]]))[0]

        pr[-out_shape[0]:, :] = pr_out.reshape((out_shape[0], out_shape[1],
                                                n_classes)).argmax(axis=2)

        binary_car_result = np.where((pr == 1), 1, 0).astype('uint8')
        binary_road_result = np.where((pr == 2), 1, 0).astype('uint8')

        answer_key[frame] = [
            encode(binary_car_result),
            encode(binary_road_result)
        ]
        # Increment frame
        frame+=1

    # Print output in proper json format
    print(json.dumps(answer_key))
