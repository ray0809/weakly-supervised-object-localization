# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:14:24 2017

@author: ray
"""
import os
import tensorflow as tf
import keras.backend as k
from keras.applications import InceptionV3,ResNet50,VGG16,VGG19
from keras.layers import Input,GlobalAveragePooling2D,Dense,Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import adam,rmsprop,sgd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
train = 'train/'
test = 'test/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

cifar_data = cifar10.load_data()
train_data = preprocess_input((cifar_data[0][0]).astype('float32'))
train_label = cifar_data[0][1]
test_data = preprocess_input((cifar_data[1][0]).astype('float32'))
test_label = cifar_data[1][1]
train_label = to_categorical(train_label,10)
test_label = to_categorical(test_label,10)


def Resize(x):
    y = tf.image.resize_bicubic(x,size=(256,256))
    return y



'''
Here we set the default input image size for this model is 224x224.



def creat_model():
    inputs = Input(shape=(32,32,3))
    #Notic : preprocess is different in each Model
    resize = Lambda(Resize,(256,256,3))(inputs)
    #normal = Lambda(preprocess_input,(256,256,3))(resize)
    base_model = VGG16(include_top=False)
    temp_model = Model(base_model.input,base_model.get_layer('block5_conv3').output)
    conv = temp_model(resize)
    #conv = base_model(resize)
    GAV = GlobalAveragePooling2D()(conv)
    outputs = Dense(10,activation='softmax')(GAV)
    model = Model(inputs,outputs)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
'''

def creat_model():
    inputs = Input(shape=(32,32,3))
    #Notic : preprocess is different in each Model
    resize = Lambda(Resize,(256,256,3))(inputs)
    #normal = Lambda(preprocess_input,(256,256,3))(resize)
    base_model = InceptionV3(include_top=False)
    conv = base_model(resize)
    #conv = base_model(resize)
    GAV = GlobalAveragePooling2D()(conv)
    outputs = Dense(10,activation='softmax')(GAV)
    model = Model(inputs,outputs)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


'''    
def training(model):
    datagen = ImageDataGenerator()
    train_gen = datagen.flow_from_directory('train',target_size=(224,224),batch_size=4)
    test_gen = datagen.flow_from_directory('test',target_size=(224,224),batch_size=4)
    earlystopping = EarlyStopping()
    model.fit_generator(train_gen,
                        samples_per_epoch=20, nb_epoch=5000,
                        verbose=1, callbacks=[earlystopping],
                        validation_data=test_gen, nb_val_samples=20)
    
    model.save('model_save_file')
'''


def training(model):
    earlystopping = EarlyStopping()
    modelchenkpoint = ModelCheckpoint('model_best_only',save_best_only=True)
    model.fit(train_data,train_label,
              batch_size=25,
              nb_epoch=10,
              validation_data=(test_data,test_label),
              callbacks=[earlystopping,modelchenkpoint])

    #model.save('model_save_file.h5')
  



def get_conv(model,test_imgs):
    #Using InceptionV3's output
    #base_model = Model(model.input,model.get_output_at.output)
    inputs = Input(shape=(32,32,3))
    resize = Lambda(Resize,(256,256,3))(inputs)
    inception_v3 = model.get_layer('inception_v3')
    outputs = inception_v3(resize)
    new_model = Model(inputs,outputs)
    print('Loading the conv_features of test_images .......')
    conv_features = new_model.predict(test_imgs)
    print('Loading the conv_features done!!!')
    return conv_features

if __name__ == '__main__':
    model = creat_model()
    if not os.path.exists('model_save_file.h5'):
        training(model)
    else:
        model.load_weights('model_save_file.h5')
    
    conv_features = get_conv(model,test_data)
    print('Predict the labels of test_images .......')
    predict_label = model.predict(test_data)
    print('Extraction the weight between GAV and dense(2048x10) .......')
    w = model.get_weights()[-2]
    
    
