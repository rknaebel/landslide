import dataset
import h5dataset


def test_create_h5dataset():
    path = "/tmp/data.h5"
    files = "data/"
    h5dataset.make_dataset(files, path)
    sat_images = dataset.load_sat_images(files)
    altitude, slope = dataset.load_static_data(files)

    gen = h5dataset.patch_generator_from_h5(path, sat_images, altitude, slope, 25, 256, 0.4)
    next(gen)


def test_load_existing_h5dataset():
    path = "/tmp/data.h5"
    files = "data/"
    sat_images = dataset.load_sat_images(files)
    altitude, slope = dataset.load_static_data(files)

    gen = h5dataset.patch_generator_from_h5(path, sat_images, altitude, slope, 25, 256, 0.4)
    next(gen)
