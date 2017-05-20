# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
import logging

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


def get_training_image_names(test_year):
    return satellite_images[:test_year] + satellite_images[test_year + 1:]


def load_satellite_img(path, date, normalize=True):
    img = io.imread(path + date + ".tif").astype(np.float32)
    ndvi = io.imread(path + date + "_NDVI.tif").astype(np.float32)[..., None]
    if normalize:
        img /= 20000.0
        ndvi /= 255.0  # TODO ask paul: too high ?
    return img, ndvi


def load_satellite_mask(path: str, date: str):
    return io.imread(path + date + "_mask_ls.tif").astype(np.bool)


def load_static_data(path: str, normalize: bool = True):
    altitude = io.imread(path + alt).astype(np.float32)[..., None]
    slope = io.imread(path + slp).astype(np.float32)[..., None]
    if normalize:
        altitude /= 2555.0
        slope /= 52.0
    return altitude, slope


def load_image_eval(path):
    altitude, slope = load_static_data(path)
    img1 = get_single_satellite_features(path, satellite_images[-1])
    img2 = get_single_satellite_features(path, satellite_images[-2])
    return np.concatenate((img1, img2, altitude, slope), 2)


def load_mask_eval(path):
    return load_satellite_mask(path, satellite_images[-1])


def get_single_satellite_features(path, date):
    sat_image, ndvi = load_satellite_img(path, date)
    return np.concatenate((sat_image, ndvi), axis=2)


def extract_patch(data, x, y, size):
    """Expects a 3 dimensional image (height,width,channels)"""
    diff = size // 2
    patch = data[x - diff:x + diff + 1, y - diff:y + diff + 1, :]
    return patch


def patch_validator(shape, pos, size):
    if ((pos[0] < size) or
            (pos[1] < size) or
            (shape[0] - pos[0] < size) or
            (shape[1] - pos[1] < size)):
        return False
    return True


def compute_coordinates(masks):
    """Expects a list of image masks and computes two sets of coordinates, one for positive events and one for
    negatives """
    positives, negatives = [], []
    for year, mask in enumerate(masks):
        logger.info("  process mask {}".format(year))
        # positive samples
        x_pos, y_pos = np.where(mask == 1)
        d_pos = np.zeros_like(x_pos) + year
        positive = np.stack((d_pos, x_pos, y_pos)).T
        positives.append(positive)
        # negative samples
        x_neg, y_neg = np.where(mask == 0)
        d_neg = np.zeros_like(x_neg) + year
        negative = np.stack((d_neg, x_neg, y_neg)).T
        negatives.append(negative)
    # put everything together
    logger.info("concatenate coordinates")
    positives = np.concatenate(positives)
    negatives = np.concatenate(negatives)
    
    return positives, negatives


def load_sat_images(path, test_image):
    sat_images = []
    for sat_image, ndvi in (load_satellite_img(path, d) for d in get_training_image_names(test_image)):
        sat_images.append(np.concatenate((sat_image, ndvi), axis=2))
    return np.stack(sat_images, axis=0)


def make_small_dataset(path, test_image):
    """Computes full dataset"""
    logger.info("load landslides and masks")
    sat_images = load_sat_images(path, test_image)
    
    logger.info("calculate coordinates per mask")
    masks = list(load_satellite_mask(path, d) for d in get_training_image_names(test_image))
    positives, negatives = compute_coordinates(masks)
    
    altitude, slope = load_static_data(path)
    
    return sat_images, positives, negatives, altitude, slope


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
                # restart generator randomly after one batch
                if np.random.rand() < 0.001:
                    break


def patch_generator(images, pos, neg, altitude, slope, size=25, batch_size=64, p=0.4):
    # calculate the batch size per label
    batch_size_pos = max(1, int(batch_size * p))
    batch_size_neg = batch_size - batch_size_pos
    image_size = images.shape[1:]
    # init index generators
    idx_pos = index_generator(pos, patch_validator, image_size, size, batch_size_pos)
    idx_neg = index_generator(neg, patch_validator, image_size, size, batch_size_neg)
    
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


# TODO: implement following functionality
# featurewise_center=False,  # set input mean to 0 over the dataset
# samplewise_center=False,  # set each sample mean to 0
# featurewise_std_normalization=False,  # divide inputs by std of the dataset
# samplewise_std_normalization=False,  # divide each input by its std
# zca_whitening=False,  # apply ZCA whitening
# rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
# width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
# height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
# horizontal_flip=True,  # randomly flip images
# vertical_flip=False)  # randomly flip images
def augmented_patch_generator(g):
    """Expects a patch generator g and returns another generator that augments the results of g"""
    return g
