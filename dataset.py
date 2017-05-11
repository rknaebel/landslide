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


def load_satellite_mask(path, date):
    return io.imread(path + date + "_mask_ls.tif").astype(np.bool)


def load_static_data(path, normalize=True):
    altitude = io.imread(path + alt).astype(np.float32)
    slope = io.imread(path + slp).astype(np.float32)
    if normalize:
        altitude /= 2555.0
        slope /= 52.0
    return altitude, slope


# TODO bullshit!
def loadEvaluationImage(path):
    return loadSateliteFile(path, satellite_images[-1])


def extract_patch(data, x, y, size):
    """Expects a 3 dimensional image (height,width,channels)"""
    diff = size // 2
    patch = data[x - diff:x + diff + 1, y - diff:y + diff + 1, :]
    return patch


def getLandslideDataFor(date):
    altitude, slope = load_static_data(fld)
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


# TODO delete
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


def compute_coordinates(masks):
    """Expects a list of image masks and computes two sets of coordinates, one for positive events and one for
    negatives """
    positives, negatives = [], []
    for year, mask in enumerate(masks):
        logger.info("  process mask {}".format(year))
        
        logger.info("- pos")
        x_pos, y_pos = np.where(mask == 1)
        d_pos = np.zeros_like(x_pos) + year
        positive = np.stack((d_pos, x_pos, y_pos)).T
        positives.append(positive)
        
        logger.info("- neg")
        x_neg, y_neg = np.where(mask == 0)
        d_neg = np.zeros_like(x_neg) + year
        negative = np.stack((d_neg, x_neg, y_neg)).T
        negatives.append(negative)
    
    logger.info("concatenate coordinates")
    positives = np.concatenate(positives)
    negatives = np.concatenate(negatives)
    
    return positives, negatives


def make_small_dataset(fld):
    """Computes full dataset"""
    logger.info("load landslides and masks")
    masks = []
    sat_images = []
    for sat_image, ndvi, mask in (loadSateliteFile(fld, d) for d in train_images):
        sat_images.append(np.concatenate((sat_image, np.expand_dims(ndvi, 2)), axis=2))
        masks.append(mask)
    sat_images = np.stack(sat_images, axis=0)
    
    altitude, slope = load_static_data(fld)
    
    logger.info("calculate coordinates per mask")
    positives, negatives = compute_coordinates(masks)
    
    return sat_images, positives, negatives, altitude, slope


def make_small_h5dataset(path):
    f = h5py.File(path, "w")

    logger.info("load masks into memory")
    masks = list(load_satellite_mask(fld, d) for d in train_images)
    
    logger.info("calculate coordinates per mask")
    positives, negatives = compute_coordinates(masks)
    
    f.create_dataset("pos", data=positives)
    f.create_dataset("neg", data=negatives)
    
    return True


def index_generator(data, validator, image_size, size, batch_size):
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
    idx_pos = index_generator(data["pos"], patchValidator, image_size, size, batch_size_pos)
    idx_neg = index_generator(data["neg"], patchValidator, image_size, size, batch_size_neg)

    for sample_idx_pos, sample_idx_neg in zip(idx_pos, idx_neg):
        #
        X = np.stack([
            *map(lambda x: extract_patch(data["sat_images"][x[0]], x[1], x[2], size), sample_idx_pos),
            *map(lambda x: extract_patch(data["sat_images"][x[0]], x[1], x[2], size), sample_idx_neg)
        ])
        #
        y = np.concatenate((
            np.ones(batch_size_pos, dtype=np.float32),
            np.zeros(batch_size_neg, dtype=np.float32)
        ))
        yield X, y


def patch_generator(images, pos, neg, altitude, slope, size=25, batch_size=64, p=0.4):
    # calculate the batch size per label
    batch_size_pos = max(1, int(batch_size * p))
    batch_size_neg = batch_size - batch_size_pos
    image_size = images.shape[1:]
    # init index generators
    idx_pos = index_generator(pos, patchValidator, image_size, size, batch_size_pos)
    idx_neg = index_generator(neg, patchValidator, image_size, size, batch_size_neg)
    
    for sample_idx_pos, sample_idx_neg in zip(idx_pos, idx_neg):
        X = []
        for year, x, y in sample_idx_pos:
            patch_1 = extract_patch(images[year], x, y, size)
            if year == 0:
                patch_2 = patch_1
            else:
                patch_2 = extract_patch(images[year - 1], x, y, size)
            patch_atl = extract_patch(altitude, x, y, size)
            patch_slp = extract_patch(slope, x, y, size)
            X.append(np.concatenate((patch_1, patch_2, patch_atl, patch_slp), axis=2))
        
        for year, x, y in sample_idx_neg:
            patch_1 = extract_patch(images[year], x, y, size)
            if year == 0:
                patch_2 = patch_1
            else:
                patch_2 = extract_patch(images[year - 1], x, y, size)
            patch_atl = extract_patch(altitude, x, y, size)
            patch_slp = extract_patch(slope, x, y, size)
            X.append(np.concatenate((patch_1, patch_2, patch_atl, patch_slp), axis=2))
        
        X = np.stack(X)
        #
        y = np.concatenate((
            np.ones(batch_size_pos, dtype=np.float32),
            np.zeros(batch_size_neg, dtype=np.float32)
        ))
        yield X, y


# TODO use data in-memory instead of path?
# TODO story only positions into h5 file for efficiency
def patch_generator_from_small_h5(path, size=25, batch_size=64, p=0.4):
    data = h5py.File(path, "r")
    sat_images = data["sat_images"].value
    pos = data["pos"].value
    neg = data["neg"].value
    altitude = data["altitude"].value
    slope = data["slope"].value

    return patch_generator(sat_images, pos, neg, altitude, slope, size, batch_size, p)


def main():
    path = "tmp/data.h5"
    makeH5Dataset(path)
    gen = patchGeneratorFromH5(path, 25, 128, 0.4)
    for i, (X, y) in enumerate(gen):
        print(i, X.shape, y.shape)


def test():
    gen = patch_generator_from_small_h5("/tmp/data_small.h5", 25, 256, 0.4)
    for i, (X, y) in enumerate(gen):
        print(i, X.shape, y.shape)


if __name__ == "__main__":
    # main()
    pass
