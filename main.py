import argparse

from keras.models import load_model

import dataset
import evaluation
import networks

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--mode", action="store", dest="mode",
                    default="train")

parser.add_argument("--data", action="store", dest="data",
                    default="tmp/data.h5")

parser.add_argument("--model", action="store", dest="model",
                    default="models/model.h5")

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

args = parser.parse_args()

args.steps_per_epoch = args.samples // args.batch_size
args.steps_per_val = args.samples_val // args.batch_size


################################################################################


def main_train():
    print("check for data.h5")
    try:
        open(args.data, "r")
    except FileNotFoundError:
        dataset.makeH5Dataset(args.data)
    print("initialize training generator")
    train_gen = dataset.patchGeneratorFromH5(args.data,
                                             size=args.area_size,
                                             batch_size=args.batch_size,
                                             p=args.p_train)
    print("initialize validation generator")
    val_gen = dataset.patchGeneratorFromH5(args.data,
                                           size=args.area_size,
                                           batch_size=args.batch_size,
                                           p=args.p_val)
    print("get network")
    model = networks.get_model_2(args.area_size)
    print("compile")
    custom_metrics = list(evaluation.get_metrics().values())
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
    img = dataset.loadEvaluationImage()
    print("run evaluation on final year")
    y_pred = evaluation.evaluate_model(model, img, args.area_size)
    evaluation.save_image_as(y_pred, "res/out.png")

if __name__ == "__main__":
    if args.mode == "train":
        main_train()
    elif args.mode == "eval":
        main_eval()
    else:
        print("Invalid mode!")
