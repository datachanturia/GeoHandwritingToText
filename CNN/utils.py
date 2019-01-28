
from CNN.forward import *
import numpy as np


#####################################################
# ################ Utility Methods ################ #
#####################################################
def initializeFilter(size, scale=1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


def nanargmax(arr):
    # print(arr)
    idx = np.nanargmax(arr)
    # print(idx)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
    # Make predictions with trained filters/weights.
    # convolution operation
    conv1 = convolution(image, f1, b1, conv_s)
    # relu activation
    conv1[conv1 <= 0] = 0

    # second convolution operation
    conv2 = convolution(conv1, f2, b2, conv_s)
    # pass through ReLU non-linearity
    conv2[conv2 <= 0] = 0

    # maxpooling operation
    pooled = maxpool(conv2, pool_f, pool_s)
    (nf2, dim2, _) = pooled.shape
    # flatten pooled layer
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))

    # first dense layer
    z = w3.dot(fc) + b3
    # pass through ReLU non-linearity
    z[z <= 0] = 0

    # second dense layer
    out = w4.dot(z) + b4
    # predict class probabilities with the softmax activation function
    probs = softmax(out)
    
    return np.argmax(probs), np.max(probs)
