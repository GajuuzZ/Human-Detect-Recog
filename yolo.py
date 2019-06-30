#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:36:12 2019

@author: gjz
"""
import os
import cv2
import numpy as np
import colorsys
import tensorflow as tf
from tensorflow.keras.models import load_model


class YOLO(object):
    def __init__(self):
        self.model_path = 'model/yolo.h5'
        self.anchors_path = 'model/yolo_anchors.txt'
        self.classes_path = 'model/coco_classes.txt'
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.model_image_size = (416, 416) # fixed size or None.
        
        print('Model Loading...')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.yolo_model = load_model(self.model_path, compile=False)
        self._color_generate()
    
    def _color_generate(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
    
    def _padd_img(self, image, d_size):
        old_size = image.shape[:2]
    
        ratio = float(d_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
    
        rimg = cv2.resize(image, (new_size[1], new_size[0]))
    
        delta_w = d_size[0] - new_size[1]
        delta_h = d_size[0] - new_size[0]
        top, bot = delta_h//2, delta_h-(delta_h//2)
        lef, rig = delta_w//2, delta_w-(delta_w//2)
    
        pimg = cv2.copyMakeBorder(rimg,top,bot,lef,rig,cv2.BORDER_CONSTANT)
        return pimg
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def _process_feats(self, out, anchors):
        """process output features.
        # Arguments
            out: Tensor (N, N, 3, 4 + 1 +80), output feature map of yolo.
            anchors: List, anchors for box.
            mask: List, mask for anchors.
        # Returns
            boxes: ndarray (N, N, 3, 4), x,y,w,h for per box.
            box_confidence: ndarray (N, N, 3, 1), confidence for per box.
            box_class_probs: ndarray (N, N, 3, 80), class probs for per box.
        """
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])
        out = out.reshape(-1, grid_h, grid_w, len(anchors), len(self.class_names) + 5) ###

        #anchors = [anchors[i] for i in mask]
        #anchors = anchors[mask]
        #anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
        anchors_tensor = anchors.reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = self._sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.model_image_size
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs
    
    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.
        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.score)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores
    
    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep
    
    def _yolo_out(self, outs, shape):
        """Process output of yolo base net.
        # Argument:
            outs: output of yolo base net.
            shape: shape of original image.
        # Returns:
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """
        anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        #anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
        #           [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, anchors_masks):
            b, c, s = self._process_feats(out, self.anchors[mask])
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # Scale boxes back to original image shape.
        width, height = shape[1], shape[0]
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return [], [], []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores
    
    def predict(self, image, filters=None):
        """Detect the objects with yolo.
        # Arguments
            image: ndarray, processed input image.
        # Returns
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """
        
        old_shape = image.shape
        image = cv2.resize(image, self.model_image_size,
                           interpolation=cv2.INTER_CUBIC)
        image = image.astype('float32')
        image /= 255.
        image = np.expand_dims(image, 0)
        
        outs = self.yolo_model.predict(image)
        boxes, classes, scores = self._yolo_out(outs, old_shape)
        
        if filters is not None:
            idx_cls = np.where(np.isin(self.class_names, filters))[0]
            idx = np.where(np.isin(classes, idx_cls))[0]
            if boxes is not None and len(idx) > 0:
                boxes = boxes[idx]
                classes = classes[idx]
                scores = scores[idx]
            else:
                boxes = classes = scores = []
        
        return boxes, classes, scores
    
    def draw_boxes(self, image, boxes, classes, scores, filters=None):
        if boxes is None:
            return image
        for i, (box, clss, scr) in enumerate(zip(boxes, classes, scores)):
            clss_name = self.class_names[clss]
            if filters is not None and clss_name not in filters:
                continue
            
            x, y, w, h = box
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

            cv2.rectangle(image, (top, left), (right, bottom),
                          self.colors[clss], 2)
            cv2.putText(image, '{0} {1:.2f}'.format(clss_name, scr),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, self.colors[clss], 1, cv2.LINE_AA)
        
        return image

