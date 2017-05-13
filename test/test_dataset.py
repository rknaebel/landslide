import dataset


def test_call_patch_generator():
    sat_images, pos, neg, alt, slp = dataset.make_small_dataset("data/")
    gen = dataset.patch_generator(sat_images, pos, neg, alt, slp, 25, 512, 0.4)
    next(gen)


def test_patch_generator_dims():
    area = 25
    batch_size = 128
    
    sat_images, pos, neg, alt, slp = dataset.make_small_dataset("data/")
    gen = dataset.patch_generator(sat_images, pos, neg, alt, slp, area, batch_size, 0.4)
    X, y = next(gen)
    
    assert X.shape == (batch_size, area, area, 14)
