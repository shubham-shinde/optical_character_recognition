from flask import Flask, json, jsonify, request, session
import traceback, sys
from contexttimer import Timer
import time, io, math, base64
from flask_cors import CORS

import cv2
import numpy as np
# from scipy.stats import mode

import torch, torchvision
from onnxruntime import InferenceSession, SessionOptions, ExecutionMode

## INITIALISE MODEL && PARAMS INIT##
options = SessionOptions()
options.intra_op_num_threads = 32
options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
providers = ['CPUExecutionProvider']

models = [
    {
        'location': 'models/conv_v1.onnx',
        'input_shape': [1, 32, 32],
        'results': {
            'decimal': 0,
            'div': 1,
            'eight': 2,
            'equal': 3,
            'five': 4,
            'four': 5,
            'minus': 6,
            'nine': 7,
            'one': 8,
            'plus': 9,
            'seven': 10,
            'six': 11,
            'three': 12,
            'times': 13,
            'two': 14,
            'zero': 15}
    },
    {
        'location': 'models/conv_v3.onnx',
        'input_shape': [1, 32, 32],
        'results': {
            'decimal': 0,
            'div': 1,
            'eight': 2,
            'equal': 3,
            'five': 4,
            'four': 5,
            'minus': 6,
            'nine': 7,
            'one': 8,
            'plus': 9,
            'seven': 10,
            'six': 11,
            'three': 12,
            'times': 13,
            'two': 14,
            'zero': 15}
    }
]

model_index = 1
session = InferenceSession(models[model_index]['location'], providers = providers)

model_h, model_w = models[model_index]['input_shape'][1:]
## INITIALISE MODEL && PARAMS INIT ##

app = Flask(__name__)
CORS(app)

@app.route('/inference', methods=['POST'])
def get_boxes():
    response = {}
    conf_thres = 0.5
    if "conf_thres" in request.form:
        conf_thres = float(request.form["conf_thres"])
    iou_thres = 0.3
    if "iou_thres" in request.form:
        iou_thres = float(request.form["iou_thres"])
    try:
        with Timer(prefix = "read_image") as ri_timer:
            file = request.files['file']
            image_bytes = None
            if "save_file" in request.form and request.form['save_file'] == 'true':
                file_path = f'./data/server_uploaded/{time.time()}.png'
                file.save(file_path)
                image_bytes = open(file_path, 'rb').read()
            else:
                image_bytes = file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 0)
            print("image.shape", image.shape)
            response['read_image_took'] = ri_timer.elapsed

        with Timer(prefix = "model_inference") as mi_timer:
            y = model_inference(image)
            response['index'] = int(y)
            response['result'] = {v:k for k, v in models[model_index]['results'].items()}[y]
            response['model_inference_took'] = mi_timer.elapsed

    except Exception as e:
        print(e)
        response['status'] = {
            'isError': True,
            'code': 420,
            'message': str(e)
        }
        return jsonify(response)

    response['status'] = {
        'isError': False,
        'code': 200,
        'message': ""
    }
    return jsonify(response)

def model_inference(image):
    inp_resized = cv2.resize(image, (model_h, model_w))
    # inp = inp_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    inp = inp_resized[np.newaxis, :, :]
    inp = np.ascontiguousarray(inp)
    inp = inp/255
    inp = np.expand_dims(inp, axis = 0).astype('float32')
    inp = 1 - inp
    y = session.run(None, { 'input': inp })
    y = y[0]
    y = np.argmax(y, axis=1)
    return y[0]
