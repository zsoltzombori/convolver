import numpy as np
import scipy.misc
from keras import backend as K

def vis_conv_weight(W, name):
    W_scaled = W - np.min(W)
    W_scaled /= np.max(W_scaled)
    W_scaled *= 255

    if K.image_dim_ordering() == 'th':
        W_scaled = W_scaled[:, 0, :, :]
    elif K.image_dim_ordering() == 'tf':
        W_scaled = W_scaled[:, :, 0, :]
        W_scaled = np.transpose(W_scaled,(2,0,1))

    vis(W_scaled, name)


# this expects images of shape (batch, width, height)!!!
def vis(X, filename, n='default', padding=1):
    if n == 'default':
        n = int(np.sqrt(X.shape[0]))
    wx = X.shape[1]
    wy = X.shape[2]
    assert len(X) >= n*n
    X = X[:n*n]
    img = np.zeros((n*wx+(n-1)*padding, n*wy+(n-1)*padding))
    for i in range(n):
        px = i*padding
        for j in range(n):
            py = j*padding
            img[i*wx+px:(i+1)*wx+px, j*wy+py:(j+1)*wy+py] = X[i*n+j, :, :]
    img = img.clip(0, 255).astype(np.uint8)
    print("Creating image file {}".format(filename))
    scipy.misc.imsave(filename, img)
