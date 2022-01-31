import tensorflow as tf
from tensorflow import keras
import SEED

# For reproducibility
CHOSEN_SEED = SEED.seed()

tf.random.set_seed(CHOSEN_SEED)

img_height = 64
img_width = 64

kernel_weights_initializer = keras.initializers.RandomNormal(mean=0.0,
                                                             stddev=0.01)
conv_1 = keras.layers.Conv2D(8, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_1 = keras.layers.ReLU()
pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
conv_2 = keras.layers.Conv2D(16, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_2 = keras.layers.ReLU()
pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
conv_3 = keras.layers.Conv2D(32, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_3 = keras.layers.ReLU()
fully_connected = keras.layers.Dense(
    15, kernel_initializer=kernel_weights_initializer)
softmax_layer = keras.layers.Softmax()


def get_untrained_base_model():
    ####### BUILD THE BASE ARCHITECTURE, SPECIFIED IN THE FIRST PART
    # Build the architecture specified in the directions
    # Prepare the layers with the specified intializers
    ## Default bias initializer is already "zeros"

    # The input layer will also normalize the input
    base_model = keras.Sequential([
        keras.layers.Rescaling(1. / 255,
                               input_shape=(img_height, img_width, 1)),
        conv_1,
        relu_1,
        pool_1,
        conv_2,
        relu_2,
        pool_2,
        conv_3,
        relu_3,
        # Flatten before fully connected
        keras.layers.Flatten(),
        fully_connected,
        softmax_layer
    ])
    return base_model

# This is the one that works, it would seem
def aug_lr_flip(input_image, input_labels):
    new_image_batch = tf.image.flip_left_right(input_image)
    return (new_image_batch, input_labels)


# Here Batch normalization layers are added
def get_model_batchno():
    batchnomodel = keras.Sequential([
        keras.layers.Rescaling(1. / 255,
                               input_shape=(img_height, img_width, 1)),
        conv_1,
        ### BATCHNO
        keras.layers.BatchNormalization(),
        relu_1,
        pool_1,
        conv_2,
        ### BATCHNO
        keras.layers.BatchNormalization(),
        relu_2,
        pool_2,
        conv_3,
        keras.layers.BatchNormalization(),
        relu_3,
        # Flatten before fully connected
        keras.layers.Flatten(),
        fully_connected,
        softmax_layer
    ])
    return batchnomodel


def get_increasing_convmodel():
    convmodelmodel = keras.Sequential([
        keras.layers.Rescaling(1. / 255,
                               input_shape=(img_height, img_width, 1)), conv_1,
        relu_1, pool_1,
        keras.layers.Conv2D(16, (5, 5),
                            strides=(1, 1),
                            kernel_initializer=kernel_weights_initializer,
                            padding="same"), relu_2, pool_2,
        keras.layers.Conv2D(32, (7, 7),
                            strides=(1, 1),
                            kernel_initializer=kernel_weights_initializer,
                            padding="same"), relu_3,
        keras.layers.Flatten(), fully_connected, softmax_layer
    ])
    return convmodelmodel

def get_dropout_model(rates):
    dropmodel = keras.Sequential([
        keras.layers.Rescaling(1. / 255,
                               input_shape=(img_height, img_width, 1)),
        conv_1,
        # See https://keras.io/api/layers/regularization_layers/spatial_dropout2d/
        keras.layers.SpatialDropout2D(rates[0]),
        relu_1,
        pool_1,
        conv_2,
        keras.layers.Dropout(rates[1]),
        relu_2,
        pool_2,
        conv_3,
        keras.layers.Dropout(rates[2]),
        relu_3,
        # Flatten before fully connected
        keras.layers.Flatten(),
        fully_connected,
        softmax_layer
    ])
    return dropmodel


def model_l2_reg(droprate,rates):
    # Instantiate the regularizer
    l2reg = keras.regularizers.l2(l2 = droprate)
    conv_1_reg = keras.layers.Conv2D(8, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                                     padding="same", kernel_regularizer = l2reg)

    conv_2_reg = keras.layers.Conv2D(16, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                                     padding="same",kernel_regularizer = l2reg)
    conv_3_reg = keras.layers.Conv2D(32, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                                     padding="same", kernel_regularizer = l2reg)   
    regmodel = keras.Sequential([
        keras.layers.Rescaling(1. / 255,
                               input_shape=(img_height, img_width, 1)),
        conv_1_reg,
        # See https://keras.io/api/layers/regularization_layers/spatial_dropout2d/
        keras.layers.SpatialDropout2D(rates[0]),
        relu_1,
        pool_1,
        conv_2_reg,
        keras.layers.Dropout(rates[1]),
        relu_2,
        pool_2,
        conv_3_reg,
        keras.layers.Dropout(rates[2]),
        relu_3,
        # Flatten before fully connected
        keras.layers.Flatten(),
        fully_connected,
        softmax_layer
    ])
    return regmodel
