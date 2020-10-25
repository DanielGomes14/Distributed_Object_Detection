#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
#================================================================
import sys
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from core.config import cfg
from PIL import Image
from io import BytesIO
import base64
# Image to be processed

def load_weights(model, weights_file): #foi para testar o q o DG disse é do utils isto
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def detect(image_path):
    #original_image=Image.open(BytesIO(original_image)).convert("RGBA")
    original_image      = cv2.imread(image_path) #you can and should replace this line to receive the image directly (not from a file)
    #original_image = base64.b64decode(dec_image)
    # Read class names
    class_names = {}
    with open(cfg.YOLO.CLASSES, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')

    # Setup tensorflow, keras and YOLOv3
    input_size   = 416
    input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    load_weights(model, "./yolov3.weights")
    
    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    # We have our objects detected and boxed, lets move the class name into a list
    objects_detected = []
    for x0,y0,x1,y1,prob,class_id in bboxes:
        objects_detected.append(class_names[class_id])

    # Lets show the user a nice picture - should be erased in production
    #image = utils.draw_bbox(original_image, bboxes)
    #image = Image.fromarray(image)
    #image.show()
    return objects_detected
    #print(f"Objects Detected: {objects_detected}")

