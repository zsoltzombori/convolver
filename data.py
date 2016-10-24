import numpy as np
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

def load_data(dataset):
    if dataset == "cifar10":
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        nb_classes = 10
    elif dataset == "mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        nb_classes = 10
        # ImageDataGenerator requires a color axis, but is okay with it being grayscale.
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    X_train /= 255
    X_test /= 255
    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        vertical_flip=False)
    datagen.fit(X_train)

    test_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.,
        height_shift_range=0.,
        horizontal_flip=False,
        vertical_flip=False)
    test_datagen.fit(X_test)

    return (X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes
