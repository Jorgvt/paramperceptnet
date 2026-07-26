import numpy as np

def normalize_0_1(img):
    return (img-img.min())/(img.max()-img.min())

def dummy_dataset(img_size):
    img1 = np.random.normal(size=(2,*img_size))
    img2 = np.random.normal(size=(2,*img_size))
    img1, img2 = normalize_0_1(img1), normalize_0_1(img2)
    mos = np.random.normal(size=(2,))
    yield img1, img2, mos

