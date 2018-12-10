# Weakly-supervised-object-localization
  simple weakly dectector implemeted by keras and tensorflow<br>
 keras == 1.2.2<br>
 tensorflow == 0.10<br>
 
 
# dataset
cifar10(just one line): from keras.datasets import cifar10
 
# author's paper
[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)


# code
using the InceptionV3 to extract conv_features<br>
after 3 iterations,training accuracy is 99.87,testing accuracy is 95.68<br>

Step: (1):run train.py<br>
      (2):run draw_detector.py<br>

also,some tricks are learned from:https://github.com/jazzsaxmafia/Weakly_detector<br>


# Results show
Here we just show some samples...



Sample 1:
![sample and hotmat 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/1.jpg)
![combine 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/2.jpg)


Sample 2:
![sample and hotmat 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/3.jpg)
![combine 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/4.jpg)

Sample 3:
![sample and hotmat 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/5.jpg)
![combine 1](https://github.com/ray0809/weakly-supervised-object-localization/blob/master/result_pic/6.jpg)
