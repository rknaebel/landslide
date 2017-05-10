from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Lambda, Dense, Merge, Activation

from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D

def getPaulsNetwork(areaSize=8, numFilters=8):
    input_shape = (areaSize, areaSize, 5)
    conv1 = Sequential()
    conv1.add(Convolution2D(numFilters, 2, 2, activation='relu', name='conv1',
                            input_shape=input_shape, init='normal'))
    conv1.add(MaxPooling2D((1, 1), strides=(1, 1)))
    conv1.add(Dropout(0.25))
    conv1.add(Flatten(name="flatten"))

    conv2 = Sequential()
    conv2.add(Convolution2D(numFilters, 2, 2, activation='relu', name='conv2',
                            input_shape=input_shape, init='normal'))
    conv2.add(MaxPooling2D((1, 1), strides=(1, 1)))
    conv2.add(Dropout(0.25))
    conv2.add(Flatten(name="flatten"))

    slpModel = Sequential()
    # slpModel.add(Dense(1, input_shape=(1,)))
    slpModel.add(Lambda(lambda x: x + 0, input_shape=(1,)))

    altModel = Sequential()
    # altModel.add(Dense(1, input_shape=(1,)))
    altModel.add(Lambda(lambda x: x + 0, input_shape=(1,)))

    ndviModel = Sequential()
    # ndviModel.add(Dense(1, input_shape=(1,)))
    ndviModel.add(Lambda(lambda x: x + 0, input_shape=(1,)))

    model = Sequential()
    model.add(Merge([conv1, conv2, slpModel, altModel, ndviModel], mode='concat'))
    model.add(Dense(128, activation='relu', name='dense',
                    init='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))
    return model


def getConvNetLandSlideAllImage(areaSize=8, numFilter=8, dim=14):
    input_shape = (areaSize, areaSize, dim)
    model = Sequential()
    model.add(Convolution2D(numFilter, 3, 3,
                            input_shape=input_shape, init='normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(numFilter, 3, 3, init='normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten(name="flatten"))

    model.add(Dense(512, activation='relu', name='dense',
                    init='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))
    return model

def getModel1(area):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(area, area, 14)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))

    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))

    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    return model

def getModel2(area):
    model = Sequential()
    model.add(Input(shape=(area, area, 14)))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))

    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    return model