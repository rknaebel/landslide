# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import random
import h5py

import math

###############################################################################
###############################################################################
import logging

###############################################################################

# create logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

###############################################################################
###############################################################################

# global params
fld = 'data/'

sateliteImages = ['20090526', '20110514', '20120524', '20130608',
                  '20140517', '20150507', '20160526']
sateliteImages = sateliteImages
alt = 'DEM_altitude.tif'
slp = 'DEM_slope.tif'


def loadSateliteFile(date, normalize=True):
    img = io.imread(fld + date + ".tif").astype(np.float32)
    ndvi = io.imread(fld + date + "_NDVI.tif").astype(np.float32)
    mask = io.imread(fld + date + "_mask_ls.tif").astype(np.float32)
    if normalize:
        img /= 20000.0
        ndvi /= 255.0  # TODO too high ?
    return img, ndvi, mask


def loadStaticData(normalize=True):
    altitude = io.imread(fld + alt).astype(np.float32)
    slope = io.imread(fld + slp).astype(np.float32)
    if normalize:
        altitude /= 2555.0
        slope /= 52.0
    return altitude, slope


def extractPatch(data, date, x, y, size):
    diff = size // 2
    patch = data[date, x - diff:x + diff + 1, y - diff:y + diff + 1, :]
    return patch


def getLandslideDataFor(date):
    altitute, slope = loadStaticData()
    date_idx = sateliteImages.index(date)
    prev_date_idx = date_idx - 1 if date_idx >= 1 else 0
    img, ndvi, mask = loadSateliteFile(date)
    prev_img, prev_ndvi, _ = loadSateliteFile(sateliteImages[prev_date_idx])
    image = np.concatenate((img, np.expand_dims(ndvi, 2)), axis=2)
    prev_image = np.concatenate((prev_img, np.expand_dims(prev_ndvi, 2)), axis=2)
    image = np.concatenate((image, prev_image, np.expand_dims(altitute, 2), np.expand_dims(slope, axis=2)), axis=2)
    return image, mask


def patchValidator(shape, pos, size):
    if ((pos[0] < size) or
            (pos[1] < size) or
            (shape[0] - pos[0] < size) or
            (shape[1] - pos[1] < size)):
        return False
    return True


def makeH5Dataset(path):
    f = h5py.File(path, "w")

    logger.info("load landslides and masks")
    sat_images, masks = zip(*(getLandslideDataFor(d) for d in sateliteImages))
    sat_images = np.stack(sat_images, axis=0)

    f.create_dataset("sat_images", data=sat_images)
    del sat_images

    logger.info("calculate coordinates per mask")
    positives, negatives = [], []
    for year, mask in enumerate(masks):
        logger.info("  process mask {}".format(year))
        num_pos = int(mask.sum())
        num_neg = int(mask.size - num_pos)

        pos = zip(*np.where(mask == 1))
        # positives.append(np.array([(idx, x, y) for x, y in pos]))
        positive = np.empty((num_pos, 3), dtype=np.int32)
        for idx, (x, y) in enumerate(pos):
            positive[idx] = (year, x, y)
        positives.append(positive)

        neg = zip(*np.where(mask == 0))
        # negatives.append(np.array([(year, x, y) for x, y in neg]))
        negative = np.empty((num_neg, 3), dtype=np.int32)
        for idx, (x, y) in enumerate(neg):
            negative[idx] = (year, x, y)
        negatives.append(negative)

    logger.info("concatenate coordinates")
    positives = np.concatenate(positives)
    negatives = np.concatenate(negatives)

    f.create_dataset("pos", data=positives)
    f.create_dataset("neg", data=negatives)

    return True


def indexGenerator(data, validator, image_size, size, batch_size):
    """
    
    :param data: fh on h5py dataset
    :param validator: filter applied to every position
    :param batch_size: 
    :return: 
    """
    batch = np.empty((batch_size, 3), dtype=np.int32)
    ctr = 0
    while True:
        indices = np.random.permutation(len(data))
        for i in indices:
            if validator(image_size, data[i][1:], size):
                batch[ctr] = data[i]
                ctr += 1
            if ctr == batch_size:
                yield batch
                ctr = 0


def patchGeneratorFromH5(path, size=25, batch_size=64, p=0.4, years=[0]):
    data = h5py.File(path, "r")
    # calculate the batch size per label
    batch_size_pos = int(batch_size * p)
    batch_size_neg = batch_size - batch_size_pos
    image_size = data["sat_images"].shape[1:]
    # init index generators
    idx_pos = indexGenerator(data["pos"], patchValidator, image_size, size, batch_size_pos)
    idx_neg = indexGenerator(data["neg"], patchValidator, image_size, size, batch_size_neg)

    for sample_idx_pos, sample_idx_neg in zip(idx_pos, idx_neg):
        #
        X = np.stack([
            *map(lambda x: extractPatch(data["sat_images"], x[0], x[1], x[2], size), sample_idx_pos),
            *map(lambda x: extractPatch(data["sat_images"], x[0], x[1], x[2], size), sample_idx_neg)
        ])
        #
        y = np.concatenate((
            np.ones(batch_size_pos, dtype=np.float32),
            np.zeros(batch_size_neg, dtype=np.float32)
        ))
        yield X, y


if __name__ == "__main__":
    path = "tmp/data.h5"
    makeH5Dataset(path)
    gen = patchGeneratorFromH5(path, 25, 128, 0.4)
    for i, (X, y) in enumerate(gen):
        print(i, X.shape, y.shape)
