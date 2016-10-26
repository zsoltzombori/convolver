import numpy as np
from keras.layers import Input, Dense, Lambda, Reshape, Dropout, Flatten, Activation
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from convolver import *

import sys
sys.path.append('/home/zombori//k-arm')
sys.path.append('/Users/zsoltzombori/git/k-arm')
from arm import ArmLayer

weight_decay = 1e-4

def conv_model(input_shape, nb_classes, batch_size, lr, nb_filters):
    pool_size = (2,2)
    kernel_size = (3,3)
    input = Input(shape=input_shape[1:])
    output = input

    layers = []
    layers.append(Flatten())
    layers.append(Dense(nb_filters, input_shape=[np.prod(pool_size)]))
    output = ConvLayer(output, batch_size, layers, kernel_size[0], kernel_size[1], subsample=(1,1))
    output = Activation('relu')(output)

    layers = []
    layers.append(Flatten())
    layers.append(Dense(nb_filters, input_shape=[np.prod(pool_size)]))
    output = ConvLayer(output, batch_size, layers, kernel_size[0], kernel_size[1], subsample=(1,1))
    output = Activation('relu')(output)

    output = MaxPooling2D(pool_size=pool_size)(output)
    output = Dropout(0.5)(output)

    output = Flatten()(output)
    output = Dense(128, activation="relu", W_regularizer=l2(weight_decay))(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_classes, activation="softmax", W_regularizer=l2(weight_decay))(output)

    model = Model(input=input, output=output)
    optimizer = RMSprop(lr=lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# just like a standard mnist_conv net, but using conv arm layers instead
def arm_model(input_shape, nb_classes, batch_size, lr, iteration, threshold, reconsCoef, nb_filters):
    pool_size = (2,2)
    kernel_size = (3,3)
    input = Input(shape=input_shape[1:])
    output = input

    layers = []
    layers.append(Flatten())
    layers.append(ArmLayer(dict_size=nb_filters,iteration = iteration,threshold = threshold, reconsCoef = reconsCoef, name = "arm_1"))
    output = ConvLayer(output, batch_size, layers, kernel_size[0], kernel_size[1], subsample=(1,1))
    output = Activation('relu')(output)

    layers = []
    layers.append(Flatten())
    layers.append(ArmLayer(dict_size=nb_filters,iteration = iteration, threshold = threshold, reconsCoef = reconsCoef, name = "arm_2"))
    output = ConvLayer(output, batch_size, layers, kernel_size[0], kernel_size[1], subsample=(1,1))
    output = Activation('relu')(output)

    output = MaxPooling2D(pool_size=pool_size)(output)
    output = Dropout(0.5)(output)

    output = Flatten()(output)
    output = Dense(128, activation="relu", W_regularizer=l2(weight_decay))(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_classes, activation="softmax", W_regularizer=l2(weight_decay))(output)

    model = Model(input=input, output=output)
    optimizer = RMSprop(lr=lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
