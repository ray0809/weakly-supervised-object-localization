import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import InceptionV3





def Resize(x):
    y = tf.image.resize(x, size=(256,256))
    return y


def create_inceptionv3(inp_size, rz_size, class_num):
    inputs = Input(shape=(inp_size, inp_size, 3))
    resize = Lambda(Resize,(rz_size, rz_size, 3))(inputs)
    base_model = InceptionV3(include_top=False)
    conv = base_model(resize)
    GAV = GlobalAveragePooling2D()(conv)
    outputs = Dense(class_num, activation='softmax')(GAV)
    model = Model(inputs, outputs)
    return model


def create_multi_inceptionv3(inceptionv3_model, inp_size, rz_size, class_num):
    inputs = Input(shape=(inp_size, inp_size, 3))
    resize = Lambda(Resize,(rz_size, rz_size, 3))(inputs)
    
    inception_v3 = inceptionv3_model.get_layer('inception_v3')
    conv = inception_v3(resize)

    # resize, for the same size with original pic, concat for imshow
    resized_conv = Lambda(Resize,(rz_size, rz_size, 3))(conv)

    GAV = GlobalAveragePooling2D()(conv)

    dense = inceptionv3_model.get_layer('dense')
    outputs = dense(GAV)
    middle_model = Model(inputs, [resized_conv, outputs])

    # the last dense layer's weight  2048*class_num
    w = middle_model.get_layer('dense').weights[0].numpy()
    return middle_model, w


if __name__ == "__main__":
    m = create_inceptionv3(32, 256, 10)
    mm, w = create_multi_inceptionv3(m, 32, 256, 10)
    


    import numpy as np
    inp = np.zeros((1,32,32,3))
    rz_conv, output = mm(inp)
    print(rz_conv.shape)
    print(output)