# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import random
import matplotlib.pyplot as plt

import math

# global params
fld = 'data/'

sateliteImages = ['20090526', '20110514', '20120524', '20130608',
                  '20140517', '20150507', '20160526']
sateliteImages = sateliteImages[:3]
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


def loadLandslideDataset(dates):
    last_image = None  # TODO maybe replace by another flag
    satelite_images = []
    dates = [dates[0]] + dates
    for date in dates:
        img, nvdi, mask = loadSateliteFile(date)
        image = np.concatenate((img, np.expand_dims(nvdi, 2)), axis=2)
        if last_image != None:
            image = np.concatenate((image, last_image), axis=2)
            satelite_images.append((image, mask))
        else:
            last_image = image

    altitude, slope = loadStaticData()

    return satelite_images, altitude, slope


def getLandslideDataFor(date):
    date_idx = sateliteImages.index(date)
    prev_date_idx = date_idx - 1 if date_idx >= 1 else 0
    img, nvdi, mask = loadSateliteFile(date)
    prev_img, prev_nvdi, prev_mask = loadSateliteFile(sateliteImages[prev_date_idx])
    image = np.concatenate((img, np.expand_dims(nvdi, 2)), axis=2)
    prev_image = np.concatenate((prev_img, np.expand_dims(prev_nvdi, 2)), axis=2)
    image = np.concatenate((image, prev_image), axis=2)
    return image, mask


def patchValidator(shape, pos, size):
    if ((pos[0] < size) or
            (pos[1] < size) or
            (shape[0] - pos[0] < size) or
            (shape[1] - pos[1] < size)):
        return False
    return True


def getDataset():
    sat_images, masks = zip(*(getLandslideDataFor(d) for d in sateliteImages))
    sat_images = np.stack(sat_images)
    altitute, slope = loadStaticData(normalize=True)

    positives, negatives = [], []
    for idx, mask in enumerate(masks):
        pos = zip(*np.where(mask == 1))
        positives += [(idx, x, y) for x, y in pos]
        neg = zip(*np.where(mask == 0))
        negatives += [(idx, x, y) for x, y in neg]

    return sat_images, positives, negatives, altitute, slope


def LandslideGenerator(data, size=25, batch_size=64, p=0.2, normalize=True):
    sat_images, positives, negatives, altitute, slope = data

    # filter patches that satisfy the validator - coordinates where area is within the image
    positives = np.array(list(filter(lambda x: patchValidator(sat_images.shape[1:], x[1:], size), positives)))
    negatives = np.array(list(filter(lambda x: patchValidator(sat_images.shape[1:], x[1:], size), negatives)))
    # calculate the batch size per label
    batch_size_pos = int(batch_size * p)
    batch_size_neg = batch_size - batch_size_pos

    while True:
        #
        idx_pos = np.array_split(np.random.permutation(len(positives)), math.ceil(len(positives) / batch_size_pos))[:-1]
        idx_neg = np.array_split(np.random.permutation(len(negatives)), math.ceil(len(negatives) / batch_size_neg))[:-1]
        #
        for sample_idx_pos, sample_idx_neg in zip(idx_pos, idx_neg):
            #
            X = np.stack([
                *map(lambda x: extractPatch(sat_images, x[0], x[1], x[2], size), positives[sample_idx_pos]),
                *map(lambda x: extractPatch(sat_images, x[0], x[1], x[2], size), negatives[sample_idx_neg])
            ])
            #
            y = np.concatenate((
                np.ones(len(sample_idx_pos), dtype=np.float32), # TODO len(sample_idx_pos) == batch_size_pos?!
                np.zeros(len(sample_idx_neg), dtype=np.float32)
            ))
            yield X, y


def LandslideData(date, size=25, normalize=True):
    # TODO what happens with size==0? Does even numbers make sense?
    # load data
    print("load landslide for given date")
    sat_image, mask = getLandslideDataFor(date)
    altitude, slope = loadStaticData()
    # generate coordinates (one for each set of lables)
    print("create coordinates for pos and neg labels")
    positive = zip(*np.where(mask == 1))
    negative = zip(*np.where(mask == 0))
    # TODO sample ratio of p from s1 and (1-p) from s2
    # extract patches (subimages) of positive and negative samples
    correctSize = lambda x: x.shape == (size, size, 12)
    print("sample positive")
    sample_pos = list(filter(correctSize, map(lambda x: extractPatch(sat_image, x, size), positive)))
    print("sample negative")
    sample_neg = list(filter(correctSize, map(lambda x: extractPatch(sat_image, x, size), negative)))
    # combine to np batch and return
    print("stack samples")
    X = np.stack(sample_pos + sample_neg)
    print("create corresponding labels")
    y = np.concatenate((
        np.ones((len(sample_pos)), dtype=np.float32),
        np.zeros((len(sample_neg)), dtype=np.float32)
    ))
    return X, y

