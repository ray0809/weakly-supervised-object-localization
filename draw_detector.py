# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:53:09 2017

@author: ray
"""


import numpy as np
import cv2
import tensorflow as tf

W = np.zeros((10000,2048),dtype='float32')

w = np.load('weight.npy')
conv = np.load('conv_features.npy')
predict = np.load('predict_label.npy')


predict_label = (np.argmax(predict,axis=1)).astype('uint8')
#this W meas the coefficients of linear combination:10000x2048
W = np.transpose(w)[predict_label,:]
W = np.expand_dims(W,axis=2)


#W and conv are too large
temp1 = W[0:10]
temp2 = conv[0:10]

del W
del conv




sess = tf.Session()

weight = tf.placeholder(dtype='float32',shape=temp1.shape,name='weight')
features = tf.placeholder(dtype='float32',shape=temp2.shape,name='features')

resize_features = tf.image.resize_bicubic(features,size=(256,256))
resize_features = tf.reshape(resize_features,[-1,256*256,2048])

Hotmap = tf.batch_matmul(resize_features,weight)
Hotmap = tf.reshape(Hotmap,[-1,256,256])

result_map = sess.run(Hotmap,feed_dict={weight:temp1,features:temp2})

