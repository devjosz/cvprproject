import tensorflow as tf
from tensorflow import keras

# The function prefixed by f_ are to be given as argument to a
# tf.data.Dataset.map() function, they are not suitable as
# standalones.
def f_gray_2rgb_prep(input_image,input_labels):
    new_image_batch = tf.image.grayscale_to_rgb(input_image)
    return (new_image_batch,input_labels)

def f_resnet50_pre(input_image,input_labels):
    proced_image = keras.applications.resnet50.preprocess_input(input_image)
    return (proced_image,input_labels)
    
def f_vgg16_pre(input_image, input_labels):
    proced_image = keras.applications.vgg16.preprocess_input(input_image)
    return (proced_image,input_labels)

def f_vgg19_pre(input_image,input_labels):
    proced_image = keras.applications.vgg19.preprocess_input(input_image)
    return (proced_image,input_labels)