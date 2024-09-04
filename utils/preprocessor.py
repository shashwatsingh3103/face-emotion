import numpy as np
from PIL import Image

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(image_name):
    with Image.open(image_name) as img:
        return np.array(img)

def _imresize(image_array, size):
    img = Image.fromarray(image_array)
    img_resized = img.resize(size, Image.ANTIALIAS)
    return np.array(img_resized)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
