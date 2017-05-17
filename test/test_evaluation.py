import numpy as np

import evaluation


def test_generate_patches_full():
    img = np.random.rand(3, 3, 1)
    x, y, z = img.shape
    
    img = evaluation.padding(img, 1)
    
    g = evaluation.generate_patches_full(img, 1, 3)
    res = np.concatenate(list(g), axis=0)
    assert res.shape[0] == x * y
    
    g = evaluation.generate_patches_full(img, 2, 3)
    res = np.concatenate(list(g), axis=0)
    assert res.shape[0] == x * y
    
    g = evaluation.generate_patches_full(img, 7, 3)
    res = np.concatenate(list(g), axis=0)
    assert res.shape[0] == x * y
