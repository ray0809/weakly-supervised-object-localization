import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow.keras.applications.inception_v3 import preprocess_input
from model import create_inceptionv3



def train():
    
    # Load Dataset
    cifar_data = cifar10.load_data()
    train_data = preprocess_input((cifar_data[0][0]).astype('float32'))
    train_label = cifar_data[0][1]
    test_data = preprocess_input((cifar_data[1][0]).astype('float32'))
    test_label = cifar_data[1][1]
    train_label = to_categorical(train_label,10)
    test_label = to_categorical(test_label,10)






    # Init Classifier
    inception_model = create_inceptionv3(32, 256, 10)
    inception_model.compile(optimizer='sgd',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    
    
    # Traing Classifier
    if not os.path.isdir('./checkpoint'):
        os.makedirs('./checkpoint')
    earlystopping = EarlyStopping(patience=2)
    modelchenkpoint = ModelCheckpoint('./checkpoint/best.h5', 
                                      save_best_only=True,
                                      save_weights_only=True)
    inception_model.fit(train_data, train_label,
              batch_size=64,
              epochs=10,
              validation_data=(test_data, test_label),
              callbacks=[earlystopping, modelchenkpoint])

 


if __name__ == "__main__":
    train()