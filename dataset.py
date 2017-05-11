# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
import logging

import h5py
import numpy as np
from skimage import io

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

satellite_images = ['20090526', '20110514', '20120524', '20130608',
                    '20140517', '20150507', '20160526']
train_images = satellite_images[:-1]
alt = 'DEM_altitude.tif'
slp = 'DEM_slope.tif'


def loadSateliteFile(path, date, normalize=True):
    img = io.imread(path + date + ".tif").astype(np.float32)
    ndvi = io.imread(path + date + "_NDVI.tif").astype(np.float32)
    mask = io.imread(path + date + "_mask_ls.tif").astype(np.float32)
    if normalize:
        img /= 20000.0
        ndvi /= 255.0  # TODO ask paul: too high ?
    return img, ndvi, mask


def loadStaticData(normalize=True):
    altitude = io.imread(fld + alt).astype(np.float32)
    slope = io.imread(fld + slp).astype(np.float32)
    if normalize:
        altitude /= 2555.0
        slope /= 52.0
    return altitude, slope


def loadEvaluationImage(path):
    return loadSateliteFile(path, satellite_images[-1])


# TODO remove date from arguments
def extractPatch(data, date, x, y, size):
    diff = size // 2
    patch = data[date, x - diff:x + diff + 1, y - diff:y + diff + 1, :]
    return patch


def getLandslideDataFor(date):
    altitude, slope = loadStaticData()
    date_idx = train_images.index(date)
    prev_date_idx = date_idx - 1 if date_idx >= 1 else 0
    img, ndvi, mask = loadSateliteFile(fld, date)
    prev_img, prev_ndvi, _ = loadSateliteFile(train_images[prev_date_idx])
    image = np.concatenate((img, np.expand_dims(ndvi, 2)), axis=2)
    prev_image = np.concatenate((prev_img, np.expand_dims(prev_ndvi, 2)), axis=2)
    image = np.concatenate((image, prev_image, np.expand_dims(altitude, 2), np.expand_dims(slope, axis=2)), axis=2)
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
    sat_images, masks = zip(*(getLandslideDataFor(d) for d in train_images))
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
        positive = np.empty((num_pos, 3), dtype=np.int32)
        for idx, (x, y) in enumerate(pos):
            positive[idx] = (year, x, y)
        positives.append(positive)

        neg = zip(*np.where(mask == 0))
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


def make_small_h5dataset(path):
    f = h5py.File(path, "w")
    
    logger.info("load landslides and masks")
    masks = []
    sat_images = []
    for sat_image, ndvi, mask in (loadSateliteFile(fld, d) for d in train_images):
        sat_images.append(np.concatenate((sat_image, np.expand_dims(ndvi, 2)), axis=2))
        masks.append(mask)
    sat_images = np.stack(sat_images, axis=0)
    
    f.create_dataset("sat_images", data=sat_images)
    del sat_images
    
    altitude, slope = loadStaticData()
    f.create_dataset("altitude", data=np.expand_dims(altitude, 2))
    del altitude
    f.create_dataset("slope", data=np.expand_dims(slope, 2))
    del slope
    
    logger.info("calculate coordinates per mask")
    positives, negatives = [], []
    for year, mask in enumerate(masks):
        logger.info("  process mask {}".format(year))
        num_pos = int(mask.sum())
        num_neg = int(mask.size - num_pos)
        
        pos = zip(*np.where(mask == 1))
        positive = np.empty((num_pos, 3), dtype=np.int32)
        for idx, (x, y) in enumerate(pos):
            positive[idx] = (year, x, y)
        positives.append(positive)
        
        neg = zip(*np.where(mask == 0))
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


def patchGeneratorFromH5(path, size=25, batch_size=64, p=0.4):
    data = h5py.File(path, "r")
    # calculate the batch size per label
    batch_size_pos = max(1, int(batch_size * p))
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


# TODO use data in-memory instead of path?
def patch_generator_from_small_h5(path, size=25, batch_size=64, p=0.4):
    data = h5py.File(path, "r")
    # calculate the batch size per label
    batch_size_pos = max(1, int(batch_size * p))
    batch_size_neg = batch_size - batch_size_pos
    sat_images = np.array(data["sat_images"])
    image_size = sat_images.shape[1:]
    # init index generators
    idx_pos = indexGenerator(data["pos"], patchValidator, image_size, size, batch_size_pos)
    idx_neg = indexGenerator(data["neg"], patchValidator, image_size, size, batch_size_neg)
    
    altitude = np.expand_dims(data["altitude"], 0)
    slope = np.expand_dims(data["slope"], 0)
    
    for sample_idx_pos, sample_idx_neg in zip(idx_pos, idx_neg):
        X = []
        for year, x, y in sample_idx_pos:
            patch_1 = extractPatch(sat_images, year, x, y, size)
            if year == 0:
                patch_2 = patch_1
            else:
                patch_2 = extractPatch(sat_images, year - 1, x, y, size)
            patch_atl = extractPatch(altitude, 0, x, y, size)
            patch_slp = extractPatch(slope, 0, x, y, size)
            X.append(np.concatenate((patch_1, patch_2, patch_atl, patch_slp), axis=2))
        
        for year, x, y in sample_idx_neg:
            patch_1 = extractPatch(sat_images, year, x, y, size)
            if year == 0:
                patch_2 = patch_1
            else:
                patch_2 = extractPatch(sat_images, year - 1, x, y, size)
            patch_atl = extractPatch(altitude, 0, x, y, size)
            patch_slp = extractPatch(slope, 0, x, y, size)
            X.append(np.concatenate((patch_1, patch_2, patch_atl, patch_slp), axis=2))
        
        X = np.stack(X)
        #
        y = np.concatenate((
            np.ones(batch_size_pos, dtype=np.float32),
            np.zeros(batch_size_neg, dtype=np.float32)
        ))
        yield X, y


def main():
    path = "tmp/data.h5"
    makeH5Dataset(path)
    gen = patchGeneratorFromH5(path, 25, 128, 0.4)
    for i, (X, y) in enumerate(gen):
        print(i, X.shape, y.shape)


if __name__ == "__main__":
    # main()
    pass
