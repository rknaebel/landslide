import keras.backend as K
from keras.layers import Lambda


# import numpy as np


def Maxout():
    return Lambda(lambda x: K.max(x, axis=3, keepdims=True))

# def store_image_prediction(img, path):
#     np.save(path, img)
