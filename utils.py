import keras.backend as K
from keras.layers import Lambda


def Maxout():
    return Lambda(lambda x: K.max(x, axis=3, keepdims=True))
