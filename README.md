# Weakly-supervised-object-localization
simple weakly dectector  

we rewrite two version:  
keras + tensorflow2  
pytorch
 
# dataset
cifar10, it can be loaded by keras or pytorch with one line
 
# author's paper
[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)


# code
using the InceptionV3 to extract conv_features<br>
after 3 iterations,training accuracy is 99.87,testing accuracy is 95.68<br>

Step:   
(1):run train.py<br>
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
