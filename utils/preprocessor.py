import numpy as np
from PIL import Image

def preprocess_input(x, v2=True):
    """
    Preprocesses the input image array for model prediction.

    Args:
    x (numpy.ndarray): Input image array.
    v2 (bool): Whether to apply the V2 preprocessing (default: True).

    Returns:
    numpy.ndarray: Preprocessed image array.
    """
    x = x.astype('float32') / 255.0
    if v2:
        x = (x - 0.5) * 2.0
    return x

def _imread(image_name):
    """
    Reads an image from file and converts it to a numpy array.

    Args:
    image_name (str): Path to the image file.

    Returns:
    numpy.ndarray: Image as a numpy array.
    """
    with Image.open(image_name) as img:
        return np.array(img)

def _imresize(image_array, size):
    """
    Resizes the input image array to the specified size.

    Args:
    image_array (numpy.ndarray): Input image array.
    size (tuple): Desired size (width, height) for resizing.

    Returns:
    numpy.ndarray: Resized image array.
    """
    img = Image.fromarray(image_array)
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img_resized)

def to_categorical(integer_classes, num_classes=2):
    """
    Converts integer class labels to one-hot encoded format.

    Args:
    integer_classes (numpy.ndarray): Array of integer class labels.
    num_classes (int): Number of classes (default: 2).

    Returns:
    numpy.ndarray: One-hot encoded array.
    """
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes), dtype='float32')
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
