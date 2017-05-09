from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D

import dataset
import networks

from evaluation import precision, recall, f1_score

TEMP_PATH = "tmp/"

SIZE = 25
BATCH_SIZE = 128
QUEUE_SIZE = 50
EPOCHS = 10
STEPS_PER_EPOCH = 100000 // BATCH_SIZE

def main2():
    print("load full data into memory")
    data = dataset.getDataset()
    print("use date inside patch generator")
    lsg = dataset.LandslideGenerator(data, SIZE, BATCH_SIZE, 0.4, True)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(SIZE, SIZE, 12)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))

    # model.add(Dense(512, activation='relu', name='dense'))
    # model.add(Dropout(0.25))

    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))

    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", precision, recall, f1_score])

    model.fit_generator(lsg,
                        steps_per_epoch=10000,
                        epochs=EPOCHS,
                        verbose=True,
                        max_q_size=QUEUE_SIZE,
                        workers=1)
    
    model.save("./model.h5")

def main():
    print("initialize patch generator")
    train_gen = dataset.patchGeneratorFromH5(TEMP_PATH + "data.h5",
                                             size=SIZE,
                                             batch_size=BATCH_SIZE,
                                             p=0.2)

    val_gen = dataset.patchGeneratorFromH5(TEMP_PATH + "data.h5",
                                           size=SIZE,
                                           batch_size=BATCH_SIZE,
                                           p=0.01)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(SIZE, SIZE, 14)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1, 1), strides=(1, 1)))
    model.add(Dropout(0.25))

    # model.add(Dense(512, activation='relu', name='dense'))
    # model.add(Dropout(0.25))

    model.add(AvgPool2D((3, 3), strides=(1, 1)))
    model.add(Flatten(name="flatten"))

    model.add(Dense(1, name='last_layer'))
    model.add(Activation('sigmoid'))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", precision, recall, f1_score])




    model.fit_generator(train_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=val_gen,
                        validation_steps=100,
                        verbose=True,
                        max_q_size=QUEUE_SIZE,
                        workers=1)

    model.save(TEMP_PATH + "model.h5")

from keras.models import load_model
import h5py
import matplotlib.pyplot as plt

def padding(x,p):
    return np.lib.pad(x,((0,0),(p,p),(p,p),(0,0)),'constant', constant_values=(0,))

def evaluate_model():
    model = load_model("model.h5", custom_objects={"precision": precision,
                                                   "recall": recall,
                                                   "f1_score": f1_score})
    data = h5py.File(TEMP_PATH + "data.h5", "r")
    img = data["sat_images"][-1:]
    pred = np.empty(img.shape[1:-1])
    img = padding(data["sat_images"][-1:], 12)

    for x in range(pred.shape[0]):
        patches = np.stack([dataset.extractPatch(img, 0, x+12, y+12, 25) for y in range(pred.shape[1])])
        pred[x] = model.predict(patches).T
        if x % 10:
            plt.imshow(pred)
            plt.show()
        print(x)


if __name__ == "__main__":
    #main()
    #evaluate_model()
    pass
