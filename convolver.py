import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras import activations
from keras.utils import np_utils 

def ConvLayer(input, batch_size, innerLayers, kernel_size=(2,2), subsample=(1,1), border_mode='valid', dim_ordering='default'):
    beforeLayer = ConvReshapeBefore(kernel_size, border_mode, subsample, dim_ordering)
    output = beforeLayer(input)
    input_shape = beforeLayer.input_shape
    output_shape = get_conv_output_shape(input_shape[1:], kernel_size, subsample, dim_ordering, border_mode)
    
    assert len(innerLayers) > 0, "You need to specify a list of inner layers to be used inside the convolution"
    for layer in innerLayers:
        output = layer(output)

    output = ConvReshapeAfter(batch_size, output_shape[0], output_shape[1], dim_ordering=dim_ordering)(output)
    return output

def get_conv_output_shape(batch_input_shape, kernel_size, subsample, dim_ordering, border_mode):
    if border_mode not in {'valid', 'same'}:
        raise Exception('Invalid border mode:', border_mode)
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        nb_features = batch_input_shape[0]
        rows = batch_input_shape[1]
        cols = batch_input_shape[2]
    elif dim_ordering == 'tf':
        nb_features = batch_input_shape[2]
        rows = batch_input_shape[0]
        cols = batch_input_shape[1]
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    output_rows = np_utils.conv_output_length(rows, kernel_size[0], border_mode, subsample[0])
    output_cols = np_utils.conv_output_length(cols, kernel_size[1], border_mode, subsample[1])
    output_features = nb_features
    return [output_rows, output_cols, nb_features]

class ConvReshapeBefore(Layer):

    def __init__(self, kernel_size,
                     border_mode='valid', subsample=(1, 1),
                     dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for convolver2D:', border_mode)
        self.kernel_size = kernel_size
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        super(ConvReshapeBefore, self).__init__(**kwargs)

        
    def build(self, input_shape):
        output_shapes = get_conv_output_shape(input_shape[1:], self.kernel_size, self.subsample, self.dim_ordering, self.border_mode)
        self.output_rows = output_shapes[0]
        self.output_cols = output_shapes[1]
        self.nb_features = output_shapes[2]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (None, self.nb_features, self.kernel_size[0], self.kernel_size[1])
        elif self.dim_ordering == 'tf':
            return (None, self.kernel_size[0], self.kernel_size[1], self.nb_features)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
    def call(self, x, mask=None):
        windows = []
        for row in range(self.output_rows):
            rowStart = row * self.subsample[0]
            rowEnd = rowStart + self.kernel_size[0]
            for col in range(self.output_cols):
                colStart = col * self.subsample[1]
                colEnd = colStart + self.kernel_size[1]

                if self.dim_ordering == 'th':
                    currentWindow = x[:, :, rowStart:rowEnd, colStart:colEnd]
                else:
                    currentWindow = x[:, rowStart:rowEnd, colStart:colEnd, :]
                windows.append(currentWindow)
        output = K.concatenate(windows, axis=0)
        return output

class ConvReshapeAfter(Layer):

    def __init__(self, batch_size, output_rows, output_cols, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.batch_size = batch_size
        self.output_rows = output_rows
        self.output_cols = output_cols
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        super(ConvReshapeAfter, self).__init__(**kwargs)

        
    def build(self, input_shape):
        assert len(input_shape) == 2, 'ConvReshapeAfter expects a 2dim array: batch_size * feature_size'
        self.nb_features = input_shape[1]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (self.batch_size, self.nb_features, self.output_rows, self.output_cols)
        elif self.dim_ordering == 'tf':
            return (self.batch_size, self.output_rows, self.output_cols, self.nb_features)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
    def call(self, x, mask=None):
        if self.dim_ordering == 'th':
            cellShape = [self.batch_size, self.nb_features, 1, 1]
            rowAxis = 2
            colAxis = 3
        elif self.dim_ordering == 'tf':
            cellShape = [self.batch_size, 1, 1, self.nb_features]
            rowAxis = 1
            colAxis = 2

        resultMatrix=[]
        for row in range(self.output_rows):
            resultRow=[]
            for col in range(self.output_cols):
                start = self.batch_size * (row*self.output_rows + col)
                end = start + self.batch_size
                currentCell = K.reshape(x[start:end,:], cellShape)
                resultRow.append(currentCell)
            resultRow2 = K.concatenate(resultRow, axis=rowAxis)
            resultMatrix.append(resultRow2)
        resultMatrix2 = K.concatenate(resultMatrix, axis=colAxis)
        return resultMatrix2
