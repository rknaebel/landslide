import argparse

import tensorflow as tf
from keras.models import load_model

import dataset
import evaluation
import h5dataset
import networks
import visualize

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--mode", action="store", dest="mode",
                    default="")

parser.add_argument("--data", action="store", dest="data",
                    default="data/")

parser.add_argument("--h5data", action="store", dest="h5data",
                    default="")

parser.add_argument("--model", action="store", dest="model",
                    default="models/model.h5")

parser.add_argument("--type", action="store", dest="model_type",
                    default="simple_conv")

parser.add_argument("--batch", action="store", dest="batch_size",
                    default=64, type=int)

parser.add_argument("--epochs", action="store", dest="epochs",
                    default=10, type=int)

parser.add_argument("--samples", action="store", dest="samples",
                    default=100000, type=int)

parser.add_argument("--samples_val", action="store", dest="samples_val",
                    default=10000, type=int)

parser.add_argument("--area", action="store", dest="area_size",
                    default=25, type=int)

parser.add_argument("--queue", action="store", dest="queue_size",
                    default=50, type=int)

parser.add_argument("--p", action="store", dest="p_train",
                    default=0.5, type=float)

parser.add_argument("--p_val", action="store", dest="p_val",
                    default=0.01, type=float)

parser.add_argument("--gpu", action="store", dest="gpu",
                    default=0, type=int)

args = parser.parse_args()

args.steps_per_epoch = args.samples // args.batch_size
args.steps_per_val = args.samples_val // args.batch_size


################################################################################


def main_train():
    if args.h5data:
        print("check for data.h5")
        try:
            open(args.h5data, "r")
        except FileNotFoundError:
            h5dataset.make_dataset(args.h5data)
        print("load remaining data")
        sat_images = dataset.load_sat_images(args.data)
        alt, slp = dataset.load_static_data(args.data)
        print("initialize training generator")
        train_gen = h5dataset.patch_generator_from_h5(args.h5data, sat_images, alt, slp,
                                                      size=args.area_size,
                                                      batch_size=args.batch_size,
                                                      p=args.p_train)
        print("initialize validation generator")
        val_gen = h5dataset.patch_generator_from_h5(args.h5data, sat_images, alt, slp,
                                                    size=args.area_size,
                                                    batch_size=args.batch_size,
                                                    p=args.p_val)
    else:
        print("load data into memory")
        sat_images, pos, neg, alt, slp = dataset.make_small_dataset(args.data)
        print("initialize training generator")
        train_gen = dataset.patch_generator(sat_images, pos, neg, alt, slp,
                                            size=args.area_size,
                                            batch_size=args.batch_size,
                                            p=args.p_train)
        print("initialize validation generator")
        val_gen = dataset.patch_generator(sat_images, pos, neg, alt, slp,
                                          size=args.area_size,
                                          batch_size=args.batch_size,
                                          p=args.p_val)
    print("get network on gpu{}".format(args.gpu))
    with tf.device("/gpu:{}".format(args.gpu)):
        model = networks.get_model_by_name(args.model_type)(args)
    print("compile")
    custom_metrics = evaluation.get_metric_functions()
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"] + custom_metrics)
    print(model.summary())
    print("start training")
    model.fit_generator(train_gen,
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=args.steps_per_val,
                        verbose=True,
                        max_q_size=args.queue_size,
                        workers=1)
    print("store model")
    model.save(args.model)


def main_eval():
    print("load specified model")
    model = load_model(args.model, custom_objects=evaluation.get_metrics())
    print("load evaluation image")
    img = dataset.load_image_eval(args.data)
    print("run evaluation on final year")
    y_pred = evaluation.predict_image(model, img, args.area_size)
    visualize.save_image_as(y_pred, "res/out.png")


def main_visualization():
    pass


if __name__ == "__main__":
    if args.mode == "train":
        main_train()
    elif args.mode == "eval":
        main_eval()
    elif args.mode == "fancy":
        main_visualization()
    else:
        print("Invalid mode!")
