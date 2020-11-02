import cv2
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.keras.applications.inception_v3 import preprocess_input



from model import *

class WeaklyLocation():
    def __init__(self, multi_model, w):
        self.multi_model = multi_model
        self.w = w

    def _getOutputs(self, inp):
        # here we get featmap before globalpooling and softmax output
        conv_feat, softmax_prob = self.multi_model.predict(inp)
        return conv_feat[0], softmax_prob[0]

    def _preprocess(self, img):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def getHeatmap(self, img):
        # once with one pic
        img = self._preprocess(img)
        conv_feat, softmax_prob = self._getOutputs(img)


        max_prob_idx = np.argmax(softmax_prob)
        w = self.w[:, max_prob_idx]
        w = w.reshape(1, 1, -1)
        
        heatmap = (conv_feat * w).sum(axis=2)
        return heatmap



if __name__ == "__main__":
    # the pretrained weight path
    h5_file = './checkpoint/best.h5'
    
    # load pretrained model 
    m = create_inceptionv3(32, 256, 10)
    m.load_weights(h5_file)
    mm, w = create_multi_inceptionv3(m, 32, 256, 10)
    net = WeaklyLocation(mm, w)

    
    # predict heatmap
    img = cv2.imread('./imgs/7.jpg', 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = net.getHeatmap(img_rgb)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    img = cv2.resize(img, (256, 256))

    # drawing heatmap
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1])

    plt.subplot(1,2,2)
    plt.imshow(img[:,:,::-1])
    plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
    plt.show()
        
