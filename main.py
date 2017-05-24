import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import load_model

import dataset
import evaluation
import h5dataset
import networks
import visualize

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--modes", action="store", dest="modes", nargs="+")

parser.add_argument("--data", action="store", dest="data",
                    default="data/")

parser.add_argument("--h5data", action="store", dest="h5data",
                    default="")

parser.add_argument("--model", action="store", dest="model",
                    default="model_x")

parser.add_argument("--pred", action="store", dest="pred",
                    default="")

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

parser.add_argument("--tmp", action="store_true", dest="tmp")

parser.add_argument("--test", action="store", dest="test_image",
                    default=6, choices=range(7), type=int)

args = parser.parse_args()

args.model_name = os.path.basename(args.model)
args.steps_per_epoch = args.samples // args.batch_size
args.steps_per_val = args.samples_val // args.batch_size
args.model_path = "{}/{}/".format("/tmp/rk" if args.tmp else "results",
                                  args.model)

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)


################################################################################


def main_train():
    if args.h5data:
        print("check for data.h5")
        try:
            open(args.h5data, "r")
        except FileNotFoundError:
            h5dataset.make_dataset(args.data, args.h5data, args.test_image)
        print("load remaining data")
        sat_images = dataset.load_sat_images(args.data, args.test_image)
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
        sat_images, pos, neg, alt, slp = dataset.make_small_dataset(args.data,
                                                                    args.test_image)
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
    print("define callbacks")
    cp = ModelCheckpoint(filepath=args.model_path + "model.h5",
                         monitor='val_f05_score',
                         verbose=False,
                         save_best_only=True)
    csv = CSVLogger('{}/train.log'.format(args.model_path))
    early = EarlyStopping(monitor='val_f05_score',
                          patience=2,
                          verbose=False)
    print("start training")
    model.fit_generator(train_gen,
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=args.steps_per_val,
                        verbose=True,
                        max_q_size=args.queue_size,
                        workers=1,
                        callbacks=[cp, csv])


def main_eval():
    print("load specified model")
    model = load_model(args.model_path + "/model.h5",
                       custom_objects=evaluation.get_metrics())
    print("load evaluation image")
    img = dataset.load_image_eval(args.data, args.test_image)
    print("run evaluation on final year")
    y_pred = evaluation.predict_image(model, img, args)
    np.save("{}/pred.npy".format(args.model_path), y_pred)


def main_visualization():
    mask = dataset.load_mask_eval(args.data, args.test_image)
    y_pred_path = args.model_path + "pred.npy"
    print("plot model")
    model = load_model(args.model_path + "model.h5",
                       custom_objects=evaluation.get_metrics())
    visualize.plot_model(model, args.model_path + "model.png")
    print("plot training curve")
    logs = pd.read_csv(args.model_path + "train.log")
    visualize.plot_training_curve(logs, "{}/train.png".format(args.model_path))
    pred = np.load(y_pred_path)
    print("plot pr curve")
    visualize.plot_precision_recall(mask, pred, "{}/prc.png".format(args.model_path))
    visualize.plot_precision_recall_curves(mask, pred, "{}/prc2.png".format(args.model_path))
    print("plot roc curve")
    visualize.plot_roc_curve(mask, pred, "{}/roc.png".format(args.model_path))
    print("store prediction image")
    visualize.save_image_as(pred, "{}/pred.png".format(args.model_path))


def main_score():
    mask = dataset.load_mask_eval(args.data, args.test_image)
    pred = np.load(args.pred)
    visualize.score_model(mask, pred)


if __name__ == "__main__":
    if "train" in args.modes:
        main_train()
    if "eval" in args.modes:
        main_eval()
    if "fancy" in args.modes:
        main_visualization()
    if "scores" in args.modes:
        main_score()
