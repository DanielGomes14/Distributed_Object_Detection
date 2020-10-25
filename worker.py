import argparse
import requests
import json
import tempfile
import time
from celery import Celery
import celerypickleconfig
import pickle
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode
import cv2
import numpy as np
from celery.bin import Option
from celery import bootsteps

app = Celery('worker', broker='pyamqp://')
app.config_from_object(celerypickleconfig)

app.user_options['worker'].add(
    Option('--server', default='127.0.0.1')
)
app.user_options['worker'].add(
    Option('--port', default='5000')
)

class CustomArgs(bootsteps.Step):
    def __init__(self, worker, server, port, **options):
        global srvr,prt
        srvr=""
        prt=""
        if type(server) == list:
            srvr = server[0]
        else:
            srvr = server
        if type(port) == list:
            prt= port[0]
        else:
            prt= port
app.steps['worker'].add(CustomArgs)
flag=True
input_size   = 416


@app.task(max_retries=3,acks_late=True, reject_on_worker_lost=True, acks_on_failure_or_timeout=False)
def recv_Img(data,img):
    global flag,model,input_size
    start_time = time.time()
    frame_id = data['frame']
    frame_proc = data['proc']
    original_image= pickle.loads(img)

    class_names = {}
    with open(cfg.YOLO.CLASSES, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')

    # Setup tensorflow, keras and YOLOv3

    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if flag:    
        input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
        feature_maps = YOLOv3(input_layer)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
        utils.load_weights(model, "./yolov3.weights")
        flag=False
    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    # We have our objects detected and boxed, lets move the class name into a list
    objects_detected = []
    for x0,y0,x1,y1,prob,class_id in bboxes:
        objects_detected.append(class_names[class_id])
    #put classes and its frequency on a dictionary 
    final_dict={x:objects_detected.count(x) for x in set(objects_detected)}

    elapsed_time = time.time() - start_time
    
    message={"frame": frame_id, 'proc': frame_proc,'classes':final_dict,'timestamp':elapsed_time}
    endpoint="http://" + srvr + ':'  + prt+"/result"
    requests.post(endpoint,json=message)
    return message

