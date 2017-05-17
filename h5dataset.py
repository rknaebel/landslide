import logging

import h5py

import dataset

logger = logging.getLogger('logger')


def make_dataset(path):
    f = h5py.File(path, "w")
    
    logger.info("load masks into memory")
    masks = list(dataset.load_satellite_mask(dataset.fld, d) for d in dataset.train_images)
    
    logger.info("calculate coordinates per mask")
    positives, negatives = dataset.compute_coordinates(masks)
    
    f.create_dataset("pos", data=positives)
    f.create_dataset("neg", data=negatives)


def patch_generator_from_h5(path, sat_images, altitude, slope, size=25, batch_size=64, p=0.4, in_memory=True):
    data = h5py.File(path, "r")
    pos = data["pos"]
    neg = data["neg"]
    if in_memory:
        pos = pos.value
        neg = neg.value
    return dataset.patch_generator(sat_images, pos, neg, altitude, slope, size, batch_size, p)
