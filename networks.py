from keras.layers import Activation, AvgPool2D, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential

from utils import Maxout


def get_convnet_landslide_all(areaSize=8):
    input_shape = (areaSize, areaSize, 14)
    model = Sequential()
    model.add(Conv2D(8, 3, 3, input_shape=input_shape, init='normal'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, 3, 3, init='normal'))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten(name="flatten"))
    #
    model.add(Dense(512, activation='relu', name='dense', init='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))
    
    return model


def get_model_1(area):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(area, area, 14)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))
    #
    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))
    #
    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    return model


def get_model_2(area):
    model = Sequential()
    model.add(Conv2D(32, (5, 1), padding="same", input_shape=(area, area, 14)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5), padding="same"))
    model.add(Maxout())
    model.add(Conv2D(32, (5, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5), padding="same"))
    model.add(Maxout())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
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
    #
    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))
    #
    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    return model


model_pool = {
    "simple_conv"       : get_model_1,
    "medium_maxout_conv": get_model_2
}


def get_model_pool():
    return model_pool
