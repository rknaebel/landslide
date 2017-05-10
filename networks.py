from keras.layers import Dropout, Flatten, Lambda, Dense, Activation
from keras.layers import Merge, Conv2D, MaxPool2D, AvgPool2D
from keras.models import Sequential

from utils import Maxout


def getPaulsNetwork(areaSize=8, numFilters=8):
    input_shape = (areaSize, areaSize, 5)
    conv1 = Sequential()
    conv1.add(Conv2D(numFilters, 2, 2, activation='relu', name='conv1',
                     input_shape=input_shape, init='normal'))
    conv1.add(MaxPool2D((1, 1), strides=(1, 1)))
    conv1.add(Dropout(0.25))
    conv1.add(Flatten(name="flatten"))

    conv2 = Sequential()
    conv2.add(Conv2D(numFilters, 2, 2, activation='relu', name='conv2',
                     input_shape=input_shape, init='normal'))
    conv2.add(MaxPool2D((1, 1), strides=(1, 1)))
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
    model.add(Conv2D(numFilter, 3, 3,
                     input_shape=input_shape, init='normal'))
    model.add(Activation('relu'))
    model.add(Conv2D(numFilter, 3, 3, init='normal'))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
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
    model.add(Activation('linear', input_shape=(area, area, 14)))
    model.add(Conv2D(32, (5, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5), padding="same"))
    model.add(Maxout())
    model.add(Conv2D(32, (5, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5), padding="same"))
    model.add(Maxout())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 3), padding="same"))
    model.add(Maxout())
    model.add(Conv2D(16, (3, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 3), padding="same"))
    model.add(Maxout())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))

    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    return model
