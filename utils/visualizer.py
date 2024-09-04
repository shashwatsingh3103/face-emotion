import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy.ma as ma
import cv2
from .inference import draw_text  # Ensure the correct import path

def make_mosaic(images, num_rows, num_cols, border=1, class_names=None):
    """
    Creates a mosaic of images arranged in a grid.

    Args:
    images (numpy.ndarray): Array of images to be included in the mosaic.
    num_rows (int): Number of rows in the mosaic.
    num_cols (int): Number of columns in the mosaic.
    border (int): Border size between images (default: 1).
    class_names (list, optional): List of class names for each image (default: None).

    Returns:
    numpy.ndarray: Mosaic image.
    """
    num_images = len(images)
    image_shape = images.shape[1:]
    mosaic_shape = (num_rows * image_shape[0] + (num_rows - 1) * border,
                    num_cols * image_shape[1] + (num_cols - 1) * border)
    mosaic = ma.masked_all(mosaic_shape, dtype=np.float32)

    for idx in range(num_images):
        row = idx // num_cols
        col = idx % num_cols
        image = np.squeeze(images[idx])
        mosaic[row * (image_shape[0] + border): row * (image_shape[0] + border) + image_shape[0],
               col * (image_shape[1] + border): col * (image_shape[1] + border) + image_shape[1]] = image
    
    if class_names is not None:
        for idx, class_name in enumerate(class_names):
            row = idx // num_cols
            col = idx % num_cols
            plt.text(col * (image_shape[1] + border) + 2, row * (image_shape[0] + border) + 12, class_name,
                     color='red', fontsize=8, fontweight='bold')
    
    return mosaic

def make_mosaic_v2(images, num_mosaic_rows=None, num_mosaic_cols=None, border=1):
    """
    Creates a mosaic of images with specified rows and columns.

    Args:
    images (numpy.ndarray): Array of images to be included in the mosaic.
    num_mosaic_rows (int, optional): Number of rows in the mosaic (default: None).
    num_mosaic_cols (int, optional): Number of columns in the mosaic (default: None).
    border (int): Border size between images (default: 1).

    Returns:
    numpy.ndarray: Mosaic image.
    """
    images = np.squeeze(images)
    num_images, img_rows, img_cols = images.shape
    if num_mosaic_rows is None and num_mosaic_cols is None:
        num_mosaic_rows = num_mosaic_cols = int(np.ceil(np.sqrt(num_images)))
    num_mosaic_pixel_rows = num_mosaic_rows * (img_rows + border)
    num_mosaic_pixel_cols = num_mosaic_cols * (img_cols + border)
    mosaic = np.empty((num_mosaic_pixel_rows, num_mosaic_pixel_cols))

    for idx in range(num_images):
        row = idx // num_mosaic_cols
        col = idx % num_mosaic_cols
        y0 = row * (img_rows + border)
        y1 = y0 + img_rows
        x0 = col * (img_cols + border)
        x1 = x0 + img_cols
        mosaic[y0:y1, x0:x1] = images[idx]
    
    return mosaic

def pretty_imshow(axis, data, vmin=None, vmax=None, cmap=None):
    """
    Displays an image with a colorbar.

    Args:
    axis (matplotlib.axes.Axes): The axis on which to display the image.
    data (numpy.ndarray): Image data to display.
    vmin (float, optional): Minimum value for color scaling (default: None).
    vmax (float, optional): Maximum value for color scaling (default: None).
    cmap (matplotlib.colors.Colormap, optional): Colormap to use (default: None).
    """
    cmap = cmap or cm.jet
    vmin = vmin or data.min()
    vmax = vmax or data.max()
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    image = axis.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(image, cax=cax)

def normal_imshow(axis, data, vmin=None, vmax=None, cmap=None, axis_off=True):
    """
    Displays an image without a colorbar.

    Args:
    axis (matplotlib.axes.Axes): The axis on which to display the image.
    data (numpy.ndarray): Image data to display.
    vmin (float, optional): Minimum value for color scaling (default: None).
    vmax (float, optional): Maximum value for color scaling (default: None).
    cmap (matplotlib.colors.Colormap, optional): Colormap to use (default: None).
    axis_off (bool): Whether to turn off the axis (default: True).
    """
    cmap = cmap or cm.jet
    vmin = vmin or data.min()
    vmax = vmax or data.max()
    image = axis.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    if axis_off:
        axis.axis('off')
    return image

def display_image(face, class_vector=None, class_decoder=None, pretty=False):
    """
    Displays an image with an optional class label.

    Args:
    face (numpy.ndarray): Image data to display.
    class_vector (numpy.ndarray, optional): Class vector to decode (default: None).
    class_decoder (list, optional): Decoder for class names (default: None).
    pretty (bool): Whether to use a pretty display (default: False).
    """
    face = np.squeeze(face)
    color_map = 'gray' if len(face.shape) < 3 else None
    plt.figure()
    if class_vector is not None:
        if class_decoder is None:
            raise ValueError('class_decoder must be provided if class_vector is not None')
        class_arg = np.argmax(class_vector)
        class_name = class_decoder[class_arg]
        plt.title(class_name)
    if pretty:
        pretty_imshow(plt.gca(), face, cmap=color_map)
    else:
        plt.imshow(face, cmap=color_map)

def draw_mosaic(data, num_rows, num_cols, class_vectors=None, class_decoder=None, cmap='gray'):
    """
    Draws a mosaic of images with optional class labels.

    Args:
    data (numpy.ndarray): Array of images to include in the mosaic.
    num_rows (int): Number of rows in the mosaic.
    num_cols (int): Number of columns in the mosaic.
    class_vectors (list, optional): List of class vectors for each image (default: None).
    class_decoder (list, optional): Decoder for class names (default: None).
    cmap (matplotlib.colors.Colormap): Colormap to use (default: 'gray').
    """
    if class_vectors is not None and class_decoder is None:
        raise ValueError('class_decoder must be provided if class_vectors is not None')

    fig, axes = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(8, 8, forward=True)
    if class_vectors is not None:
        titles = [class_decoder[np.argmax(vector)] for vector in class_vectors]

    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if idx >= len(data):
                break
            image = np.squeeze(data[idx])
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].axis('off')
            if class_vectors is not None:
                axes[row, col].set_title(titles[idx], fontsize=8)
            idx += 1
    plt.tight_layout()

if __name__ == '__main__':
    from utils.utils import get_labels
    from keras.models import load_model
    import pickle

    dataset_name = 'fer2013'
    class_decoder = get_labels(dataset_name)
    faces = pickle.load(open('faces.pkl', 'rb'))
    emotions = pickle.load(open('emotions.pkl', 'rb'))
    
    # Example of using pretty_imshow
    pretty_imshow(plt.gca(), make_mosaic(faces[:4], 2, 2), cmap='gray')
    plt.show()

    # Example of using draw_mosaic
    draw_mosaic(faces, 2, 2, emotions, class_decoder)
    plt.show()

    # Load and display CNN model weights
    model = load_model('../trained_models/emotion_models/simple_CNN.985-0.66.hdf5')
    conv1_weights = model.layers[2].get_weights()[0]
    kernel_conv1_weights = np.squeeze(conv1_weights)
    kernel_conv1_weights = np.rollaxis(kernel_conv1_weights, 2, 0)
    kernel_conv1_weights = np.expand_dims(kernel_conv1_weights, -1)
    num_kernels = kernel_conv1_weights.shape[0]
    box_size = int(np.ceil(np.sqrt(num_kernels)))
    
    plt.figure(figsize=(15, 15))
    plt.title('Conv1 Weights')
    pretty_imshow(plt.gca(), make_mosaic(kernel_conv1_weights, box_size, box_size), cmap=cm.binary)
    plt.show()
