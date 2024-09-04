import cv2
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from .preprocessor import preprocess_input

def reset_optimizer_weights(model_filename):
    with h5py.File(model_filename, 'r+') as model:
        if 'optimizer_weights' in model:
            del model['optimizer_weights']

def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # Utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, gradient):
            dtype = op.inputs[0].dtype
            guided_gradient = (gradient * tf.cast(gradient > 0., dtype) *
                               tf.cast(op.inputs[0] > 0., dtype))
            return guided_gradient

def compile_saliency_function(model, activation_layer='conv2d_7'):
    input_image = model.input
    layer_output = model.get_layer(activation_layer).output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_image)[0]
    return K.function([input_image, K.learning_phase()], [saliency])

def modify_backprop(model, name, task):
    graph = tf.get_default_graph()
    with graph.gradient_override_map({'Relu': name}):

        # Get layers that have an activation
        activation_layers = [layer for layer in model.layers if hasattr(layer, 'activation')]

        # Replace relu activation
        for layer in activation_layers:
            if layer.activation == tf.nn.relu:
                layer.activation = tf.nn.relu

        # Re-instantiate a new model
        if task == 'gender':
            model_path = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
        elif task == 'emotion':
            model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        new_model = load_model(model_path, compile=False)
    return new_model

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # Normalize tensor: center on 0., ensure std is 0.1
    x = x - x.mean()
    x = x / (x.std() + 1e-5)
    x = x * 0.1

    # Clip to [0, 1]
    x = x + 0.5
    x = np.clip(x, 0, 1)

    # Convert to RGB array
    x = x * 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compile_gradient_function(input_model, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    num_classes = model.output_shape[1]
    target_layer = lambda x: target_category_loss(x, category_index, num_classes)
    model.add(Lambda(target_layer, output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = model.layers[0].get_layer(layer_name).output
    gradients = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()],
                                   [conv_output, gradients])
    return gradient_function

def calculate_gradient_weighted_CAM(gradient_function, image):
    output, evaluated_gradients = gradient_function([image, False])
    output, evaluated_gradients = output[0, :], evaluated_gradients[0, :, :, :]
    weights = np.mean(evaluated_gradients, axis=(0, 1))
    CAM = np.ones(output.shape[0:2], dtype=np.float32)
    for weight_arg, weight in enumerate(weights):
        CAM = CAM + (weight * output[:, :, weight_arg])
    CAM = cv2.resize(CAM, (64, 64))
    CAM = np.maximum(CAM, 0)
    heatmap = CAM / np.max(CAM)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image = image - np.min(image)
    image = np.minimum(image, 255)

    CAM = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    CAM = np.float32(CAM) + np.float32(image)
    CAM = 255 * CAM / np.max(CAM)
    return np.uint8(CAM), heatmap

def calculate_guided_gradient_CAM(preprocessed_input, gradient_function, saliency_function):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    saliency = saliency_function([preprocessed_input, 0])
    gradCAM = saliency[0] * heatmap[..., np.newaxis]
    return deprocess_image(gradCAM)

def calculate_guided_gradient_CAM_v2(preprocessed_input, gradient_function, saliency_function, target_size=(128, 128)):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    heatmap = np.squeeze(heatmap)
    heatmap = cv2.resize(heatmap.astype('uint8'), target_size)
    saliency = saliency_function([preprocessed_input, 0])
    saliency = np.squeeze(saliency[0])
    saliency = cv2.resize(saliency.astype('uint8'), target_size)
    gradCAM = saliency * heatmap
    gradCAM = deprocess_image(gradCAM)
    return np.expand_dims(gradCAM, -1)

if __name__ == '__main__':
    import pickle
    faces = pickle.load(open('faces.pkl', 'rb'))
    face = faces[0]
    model_filename = '../../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
    #reset_optimizer_weights(model_filename)
    model = load_model(model_filename)

    preprocessed_input = load_image(face)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_6')
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', 'emotion')
    saliency_function = compile_saliency_function(guided_model)
    guided_gradCAM = calculate_guided_gradient_CAM(preprocessed_input, gradient_function, saliency_function)

    cv2.imwrite('guided_gradCAM.jpg', guided_gradCAM)
