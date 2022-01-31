import tensorflow as tf
import numpy as np
from tensorflow import keras
import SaveRes

N_CLASSES = 15
N_NETS = 5


batch_size = 32
img_width = 64
img_height = 64
input_image_size = (img_height,img_width)
def build_model_with_batchno(seed):
    tf.random.set_seed(seed)
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

test_dataset = keras.utils.image_dataset_from_directory(
    directory="data/test",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=input_image_size,
    interpolation="bilinear",
    crop_to_aspect_ratio=False)


N_EPOCHS = 100
# In the paper, they use the arithmetic average over random crops, so
# let's preserve the aspect ratio
preserve_aspect_ratio = True
batch_size = 32
image_width = 64
image_height = 64
input_image_size = (image_height, image_width)

# Build a dataset of larger images (suitable to later do random crops)
base_training_dataset = keras.utils.image_dataset_from_directory(
        directory="data/train",
        color_mode="grayscale",
        batch_size=batch_size,
        seed=2022,
        image_size=(256,256),
        validation_split=0.15,
        subset="training",
        interpolation="bicubic",
        crop_to_aspect_ratio=False)
validation_dataset = keras.utils.image_dataset_from_directory(
        directory="data/train",
        color_mode="grayscale",
        batch_size=batch_size,
        seed=2022,
        image_size=input_image_size,
        validation_split=0.15,
        subset="validation",
        interpolation="bilinear",
        crop_to_aspect_ratio=False)
        
import StoppingCriteria
batchno_stop_criterion = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True)

#stop_criteria = [batchno_stop_criterion, add_stop_criterion]
for j in range(1, N_NETS + 1):
    i = j + 10
    # Main loop: builds and trains N_NETS network, indipendently initialized
    tf.random.set_seed(i)
    base_training_dataset = keras.utils.image_dataset_from_directory(
        directory="data/train",
        color_mode="grayscale",
        batch_size=batch_size,
        seed=i,
        image_size=(256,256),
        validation_split=0.15,
        subset="training",
        interpolation="bicubic",
        crop_to_aspect_ratio=False)
    validation_dataset = keras.utils.image_dataset_from_directory(
        directory="data/train",
        color_mode="grayscale",
        batch_size=batch_size,
        seed=i,
        image_size=input_image_size,
        validation_split=0.15,
        subset="validation",
        interpolation="bilinear",
        crop_to_aspect_ratio=False)
    # Augment the data with random crops
    augment_function = keras.Sequential([keras.layers.RandomCrop(img_height,img_width,seed = i) ] )
    training_dataset = base_training_dataset.map(lambda x,y : (augment_function(x), y)) #lambda x,y: (keras.layers.RandomCrop(img_height, img_width, seed = i), y) )
    curr_net_base = build_model_with_batchno(i)
    # add a resizing layer because some images are of different size
    curr_net = keras.Sequential([keras.layers.Resizing(img_height, img_width, crop_to_aspect_ratio = False), curr_net_base])
    # Train with adam optimizer for 50 epochs
    #add_stop_criterion = StoppingCriteria.EarlyStoppingByAccuracy(value=0.99,
    #                                                              patience=10)
    curr_net.compile(optimizer=keras.optimizers.Adam(),
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[keras.metrics.SparseCategoricalAccuracy()])
    curr_history = curr_net.fit(training_dataset, validation_data =
                                validation_dataset, epochs = N_EPOCHS,
                                callbacks = [batchno_stop_criterion])#,add_stop_criterion])
    # Save the trained network
    filename = "ensemblediff_" + str(i) + ".hd5"
    curr_net.save(filename)
    # Evaluate and save the performance of each network
    SaveRes.eval_and_save(test_dataset,curr_net,"Loss, accuracy for network_" + str(i) + ", trained for " + str(len(curr_history.history["loss"])) + " epochs: ",filename = "ensemble_metrics_diff.txt")