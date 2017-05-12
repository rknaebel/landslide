import logging

import h5py

import dataset

logger = logging.getLogger('logger')


def make_small_h5dataset(path):
    f = h5py.File(path, "w")
    
    logger.info("load masks into memory")
    masks = list(dataset.load_satellite_mask(dataset.fld, d) for d in dataset.train_images)
    
    logger.info("calculate coordinates per mask")
    positives, negatives = dataset.compute_coordinates(masks)
    
    f.create_dataset("pos", data=positives)
    f.create_dataset("neg", data=negatives)
    
    return True


def patch_generator_from_small_h5(path, sat_images, altitude, slope, size=25, batch_size=64, p=0.4):
    data = h5py.File(path, "r")
    pos = data["pos"].value
    neg = data["neg"].value
    return dataset.patch_generator(sat_images, pos, neg, altitude, slope, size, batch_size, p)


def test():
    gen = patch_generator_from_small_h5("/tmp/data_small.h5", 25, 256, 0.4)
    for i, (X, y) in enumerate(gen):
        print(i, X.shape, y.shape)
