import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, fbeta_score, precision_recall_curve,
    roc_auc_score, roc_curve
)


def plot_precision_recall(mask, prediction, path):
    y = mask.flatten()
    y_pred = prediction.flatten()
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    decreasing_max_precision = np.maximum.accumulate(precision)[::-1]

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.plot(recall, precision, '--b')
    ax.step(recall[::-1], decreasing_max_precision, '-r')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig(path, dpi=600)
    plt.close()


def score_model(y, prediction):
    y = y.flatten()
    y_pred = prediction.flatten()

    precision, recall, thresholds = precision_recall_curve(y, y_pred)

    print(classification_report(y, y_pred.round()))
    print("Area under PR curve", auc(recall, precision))
    print("roc auc score", roc_auc_score(y, y_pred))
    print("F1 Score", fbeta_score(y, y_pred.round(), 1))
    print("F0.5 Score", fbeta_score(y, y_pred.round(), 0.5))
    

def plot_roc_curve(mask, prediction, path):
    y = mask.flatten()
    y_pred = prediction.flatten()
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.savefig(path, dpi=600)
    plt.close()

    print("roc_auc", roc_auc)


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    cm = confusion_matrix(y_true, y_pred)
    classes = [0, 1]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in ((i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_training_curve(logs, path):
    plt.clf()
    plt.plot(logs.f1_score, label="f1_score")
    plt.plot(logs.val_f1_score, label="val_f1_score")
    plt.plot(logs.f05_score, label="f05_score")
    plt.plot(logs.val_f05_score, label="val_f05_score")
    plt.xlabel('epoch')
    plt.ylabel('percentage')
    plt.legend(loc="lower left")
    plt.savefig(path, dpi=600)
    plt.close()


def save_image_as(img, path):
    plt.clf()
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(path, dpi=600)
    plt.close()


def plot_model_as(model, path):
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
