from keras.layers import Activation, AvgPool2D, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Model, Sequential

from utils import Maxout


def get_convnet_landslide_all(areaSize: int = 8) -> Model:
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


from keras.layers import Input, merge


def get_model_3(area):
    """First inception network implementation"""
    x = input_image = Input(shape=(area, area, 14))

    tower_0 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(x)
    tower_1 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(x)
    tower_1 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(x)
    tower_2 = Conv2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)
    tower_3 = MaxPool2D((3, 3), strides=(1, 1), border_mode='same')(x)
    tower_3 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)
    x = merge([tower_0, tower_1, tower_2, tower_3], mode='concat', concat_axis=3)
    x = Dropout(0.5)(x)

    tower_0 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(x)
    tower_1 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(x)
    tower_1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(x)
    tower_2 = Conv2D(32, 5, 5, border_mode='same', activation='relu')(tower_2)
    tower_3 = MaxPool2D((3, 3), strides=(1, 1), border_mode='same')(x)
    tower_3 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(tower_3)
    x = merge([tower_0, tower_1, tower_2, tower_3], mode='concat', concat_axis=3)
    x = Dropout(0.5)(x)

    x = AvgPool2D((3, 3), strides=(1, 1))(x)
    x = Flatten()(x)
    # model.add(Dropout(0.5))
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(input_image, x)


def get_model_4(area):
    """First res network implementation"""
    x = input_image = Input(shape=(area, area, 14))

    x = Conv2D(64, 1, 1, border_mode='same')(x)

    y = Conv2D(64, 3, 1, border_mode='same')(x)
    y = Activation('relu')(y)
    y = Conv2D(64, 1, 3, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(64, 3, 1, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(64, 1, 3, border_mode='same')(y)
    # this returns x + y.
    x = merge([x, y], mode='sum')
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    y = Conv2D(64, 3, 1, border_mode='same')(x)
    y = Activation('relu')(y)
    y = Conv2D(64, 1, 3, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(64, 3, 1, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(64, 1, 3, border_mode='same')(y)
    # this returns x + y.
    x = merge([x, y], mode='sum')
    x = Activation('relu')(x)
    x = Conv2D(32, 1, 1, border_mode='same')(x)

    y = Conv2D(32, 3, 1, border_mode='same')(x)
    y = Activation('relu')(y)
    y = Conv2D(32, 1, 3, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(32, 3, 1, border_mode='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(32, 1, 3, border_mode='same')(y)
    # this returns x + y.
    x = merge([x, y], mode='sum')
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = AvgPool2D((3, 3), strides=(1, 1))(x)
    x = Flatten()(x)
    # model.add(Dropout(0.5))
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(input_image, x)


model_pool = {
    "simple_conv"       : get_model_1,
    "medium_maxout_conv": get_model_2,
    "inception_net"     : get_model_3,
    "resnet"            : get_model_4
}


def get_model_pool():
    return model_pool


def get_model_by_name(name):
    return model_pool[name]
