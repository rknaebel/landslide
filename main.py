from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D

import numpy as np
import dataset
import networks

from evaluation import precision, recall, f1_score

################################################################################

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--data", action="store", dest="data",
                    default="tmp/data.h5")

parser.add_argument("--model", action="store", dest="model",
                    default="models/model.h5")

parser.add_argument("--batch", action="store", dest="batch_size",
                    default=64, type=int)

parser.add_argument("--epochs", action="store", dest="epochs",
                    default=10, type=int)

parser.add_argument("--samples", action="store", dest="samples",
                    default=4, type=int)

parser.add_argument("--area", action="store", dest="area_size",
                    default=25, type=int)

parser.add_argument("--queue", action="store", dest="queue_size",
                    default=50, type=int)

args = parser.parse_args()

args.steps_per_epoch = args.samples // args.batch_size

################################################################################

def main2():
    print("load full data into memory")
    data = dataset.getDataset()
    print("use date inside patch generator")
    lsg = dataset.LandslideGenerator(data, args.area_size, args.batch_size, 0.4, True)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(args.area_size, args.area_size, 12)))
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
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs,
                        verbose=True,
                        max_q_size=args.queue_size,
                        workers=1)
    
    model.save("./model.h5")

def main():
    print("initialize patch generator")
    train_gen = dataset.patchGeneratorFromH5(args.data,
                                             size=args.area_size,
                                             batch_size=args.batch_size,
                                             p=0.2)

    val_gen = dataset.patchGeneratorFromH5(args.data,
                                           size=args.area_size,
                                           batch_size=args.batch_size,
                                           p=0.01)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(args.area_size, args.area_size, 14)))
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
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=100,
                        verbose=True,
                        max_q_size=args.queue_size,
                        workers=1)

    model.save(args.model)

from keras.models import load_model
import evaluation

def evaluate_model():
    model = load_model("model.h5", custom_objects={"precision": precision,
                                                   "recall": recall,
                                                   "f1_score": f1_score})
    evaluation.evaluate_model(model, 25)

if __name__ == "__main__":
    #main()
    #evaluate_model()
    pass
