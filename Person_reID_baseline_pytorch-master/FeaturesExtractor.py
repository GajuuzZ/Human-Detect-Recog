#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:03:18 2019

@author: gjz
"""
import numpy as np
import cv2
import tensorflow as tf

class ImageEncoder(object):
    def __init__(self, model_file, input_name='images',
                 output_name='features'):
        
        with tf.gfile.GFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='net')
        self.input = self.graph.get_tensor_by_name('net/%s:0' % input_name)
        self.output = self.graph.get_tensor_by_name('net/%s:0' % output_name)
        
        assert len(self.output.get_shape()) == 2
        assert len(self.input.get_shape()) == 4
        self.feature_dim = self.output.get_shape().as_list()[-1]
        self.image_shape = self.input.get_shape().as_list()[1:]
        
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=config)
        
    def extract_image_patch(self, image, bboxs, patch_shape):
        # Extract image patch from bounding box.
        patchs = np.zeros((len(bboxs), patch_shape[0], patch_shape[1], 3),
                          dtype=np.uint8)
        for i, box in enumerate(bboxs):
            x, y, w, h = box
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
            
            img = image[left:bottom, top:right]
            patchs[i] = cv2.resize(img, tuple(patch_shape[::-1]))
        
        return patchs
    
    def encode(self, image, bboxs):
        if len(bboxs) == 0: return np.array([])
        patchs = self.extract_image_patch(image, bboxs, self.image_shape[:2])
        outs = self.sess.run(self.output, feed_dict={self.input:patchs})
        
        return outs
    
    def encodeImage(self, image):
        img = cv2.resize(image, tuple(self.image_shape[:2][::-1]))
        img = np.expand_dims(img, 0)
        out = self.sess.run(self.output, feed_dict={self.input:img})
        
        return out
