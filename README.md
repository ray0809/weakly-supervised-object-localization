# Weakly-supervised-object-localization
 simple weakly dectector implemeted by keras and tensorflow<br>
 keras == 1.2.2<br>
 tensorflow == 0.10<br>
 
  
# dataset
cifar10(just one line): from keras.datasets import cifar10
 
# author's paper
[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)


# code
using the InceptionV3 to extract conv_features
after 3 iterations,training accuracy is 99.87,testing accuracy is 95.68

also,some tricks are learned from:https://github.com/jazzsaxmafia/Weakly_detector


# Results show
Here we just show some samples...



epoch 1:
![sample 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/1.jpg,https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/2.jpg)

epoch 100:
![epoch_100](https://github.com/ray0809/pix2pix/blob/master/target2pic/pic_100.png)

epoch_200:
![epoch_200](https://github.com/ray0809/pix2pix/blob/master/target2pic/pic_200.png)
