import keras.backend as K
import numpy as np

import dataset


def padding(x, p):
    return np.lib.pad(x, ((p, p), (p, p), (0, 0)), 'constant', constant_values=(0,))


def generate_patches_full(img, batch_size, size):
    x, y, z = img.shape
    offset = size // 2
    ctr = 0
    batch = np.empty((batch_size, size, size, z), dtype=np.float32)
    
    for i, j in ((i, j) for i in range(x - 2 * offset) for j in range(y - 2 * offset)):
        batch[ctr] = dataset.extract_patch(img, i + offset, j + offset, size)
        ctr += 1
        if ctr == batch_size:
            yield batch
            ctr = 0
    yield batch[:ctr]


def predict_image(model, img, args):
    offset = args.area_size // 2
    x, y, z = img.shape
    
    img = padding(img, offset)
    
    y_pred = []
    for batch in generate_patches_full(img, args.batch_size*10, args.area_size):
        y_pred.append(model.predict(batch, batch_size=args.batch_size, verbose=True))
    y_pred = np.concatenate(y_pred, axis=0)
    # reshape to original xy-dimensions
    y_pred = y_pred.reshape((x, y))
    
    return y_pred


###############################################################################


def get_metrics():
    return dict([
        ("precision", precision),
        ("recall", recall),
        ("f1_score", f1_score),
        ("f05_score", f05_score),
    ])


def get_metric_functions():
    return [precision, recall, f1_score, f05_score]


def precision(y_true, y_pred):
    # Count positive samples.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    # Count positive samples.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    return f_score(1)(y_true, y_pred)


def f05_score(y_true, y_pred):
    return f_score(0.5)(y_true, y_pred)


def f_score(beta):
    def _f(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        
        bb = beta ** 2
        
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        
        return fbeta_score
    
    return _f
