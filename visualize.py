import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

# TODO: ROC Curve analysis
# TODO: AUC


def plot_precision_recall(mask, prediction, path):
    y = mask.flatten()
    y_pred = prediction.flatten()
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    plt.plot(recall, precision)
    plt.savefig(path, dpi=600)
    plt.close()
    

def plot_roc_curve(mask, prediction, path):
    y = mask.flatten()
    y_pred = prediction.flatten()
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.savefig(path, dpi=600)
    plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="heat"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


def save_image_as(img, path):
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(path, dpi=600)
    plt.close()
