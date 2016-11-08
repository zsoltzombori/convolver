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


weight_decay = 1e-4

def build_model(input_shape, nb_classes, batch_size, kernel_size, pool_size, firstLayers, secondLayers):
    input = Input(shape=input_shape[1:])
    output = input
    output = ConvLayer(output, batch_size, firstLayers, kernel_size)
    output = Activation('relu')(output)

    output = ConvLayer(output, batch_size, secondLayers, kernel_size)
    output = Activation('relu')(output)

    output = MaxPooling2D(pool_size=pool_size)(output)
    output = Dropout(0.25)(output)

    output = Flatten()(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_classes)(output)
    output = Activation("softmax")(output)

    model = Model(input=input, output=output)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    
