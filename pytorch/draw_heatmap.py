import cv2
import torch
import numpy as np
import matplotlib.pylab as plt

import torchvision.transforms as transforms
from model import ResNet50Net



class WeaklyLocation():
    def __init__(self, net):
        self.net = net
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.net = self.net.to(self.device)
        self.net.eval()

        # 2048 * 10
        self.w = self.net.resnet.fc.weight.data.cpu().numpy().T

    def _getOutputs(self, inp):
        # here we get featmap before globalpooling and softmax output
        conv_feat, softmax_prob = self.net.inference(inp)
        return conv_feat[0], softmax_prob[0]

    def _preprocess(self, img):
        img = self.trans(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return img

    def getHeatmap(self, img):
        # once with one pic
        img = self._preprocess(img)
        with torch.no_grad():
            conv_feat, softmax_prob = self._getOutputs(img)
            conv_feat = conv_feat.data.cpu().numpy()
            softmax_prob = softmax_prob.data.cpu().numpy()

        conv_feat = conv_feat.transpose(1,2,0)
        max_prob_idx = np.argmax(softmax_prob)
        w = self.w[:, max_prob_idx]
        w = w.reshape(1, 1, -1)
        
        heatmap = (conv_feat * w).sum(axis=2)
        return heatmap



if __name__ == "__main__":
    rz_size = 224
    num_class = 10
    m = ResNet50Net(rz_size, num_class)
    m.load_state_dict(torch.load('./checkpoint/resnet50.pkl', map_location='cpu'))
    net = WeaklyLocation(m)



    # predict heatmap
    img = cv2.imread('./imgs/0.jpg', 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = net.getHeatmap(img_rgb)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    img = cv2.resize(img, (rz_size, rz_size))

    # drawing heatmap
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1])

    plt.subplot(1,2,2)
    plt.imshow(img[:,:,::-1])
    plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
    plt.show()
