# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:52:17 2017

@author: ray
"""

import cv2
import os
import glob
import shutil

label_name_path = 'Stanford40/ImageSplits/actions.txt'
image_path = 'Stanford40/JPEGImages/'
train_test_split = 'Stanford40/ImageSplits/'

def read_label():
    label = []
    with open(label_name_path) as f:
        for i in f.readlines():
            label.append(i.split()[0])
    return label[1:]

def copy_data(category,save_path,mode='train'):
    with open(train_test_split + category + '_' + mode + '.txt') as f:
        for i in f.readlines():
            img = i.split()[0]
            shutil.copy(image_path + img,save_path)
       
    
    
'''
Method : copy
'''
def creat_categories_folders():
    label = read_label()
    for i in label:
        if not os.path.exists('train/' + i):
            os.makedirs('train/' + i)
        copy_data(i,'train/' + i)
        if not os.path.exists('test/' + i):
            os.makedirs('test/' + i)
        copy_data(i,'test/' + i,mode='test')
    

'''
Method : symlink

def creat_categories_folders():
    pass
'''

   
   
   
   
   
   
  
   
   
    
'''
if __name__ == '__main__':
    creat_categories_folders()
'''