from keras.models import load_model
import keras.backend as K

import h5py
import matplotlib.pyplot as plt
import numpy as np

import dataset

TEMP_PATH = "tmp/"

def padding(x,p):
    return np.lib.pad(x,((0,0),(p,p),(p,p),(0,0)),'constant', constant_values=(0,))

def evaluate_model(model, size):
    data = h5py.File(TEMP_PATH + "data.h5", "r")
    img = data["sat_images"][-1:]
    pred = np.empty(img.shape[1:-1])
    img = padding(data["sat_images"][-1:], 12)

    offset = size // 2

    for x in range(pred.shape[0]):
        patches = np.stack([dataset.extractPatch(img, 0, x+offset, y+offset, size) for y in range(pred.shape[1])])
        pred[x] = model.predict(patches).T
        if x % 100:
            plt.imshow(pred)
            plt.show()
        print(x)



######################################################

def precision(y_true, y_pred):
    # Count positive samples.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    # Count positive samples.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    beta = 1  # fmeasure
    bb = beta ** 2

    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score