import tensorflow as tf
import numpy as np
from tensorflow import keras

N_NETS = 5
N_CLASSES = 15

image_width = 64
image_height = 64
input_image_size = (image_height,image_width)

# Build the test dataset with batch size =1 to make it easier to
# average the prediction (see the function average_nets)
test_dataset = keras.utils.image_dataset_from_directory(
    directory="data/test",
    color_mode="grayscale",
    batch_size=1,
    image_size=input_image_size,
    interpolation="bilinear",
    crop_to_aspect_ratio=False)


def average_nets(input_data, nets):
    """ Given a single input tensor (of batch_size 1 and appropriate shape for the networks), this will return a tensor containg the prediction (tensor of probabilities) """
    ### Note: only one single input (batches of size 1!!!) This function will fail otherwise. The error will stop execution
    pred_tensor_sum = np.zeros(N_CLASSES)
    for i in range(len(nets)):
        # Compute the prediction (it is a tensor containing the
        # confidence in each prediction (so in our case an array of 15
        # floating point numbers)
        pred_tensor_sum = pred_tensor_sum + nets[i].predict(input_data)
    average = (1 / N_CLASSES) * pred_tensor_sum
    pred_average_label = average.argmax()
    return average
################# PREDICT ########################
#Now load the networks and compute the loss
nets = []
for j in range(0, N_NETS):
    i = j + 10
    nets.append(keras.models.load_model("ensemblediff_" + str(i + 1) + ".hd5"))
# Predict:
total_correct = 0
counter = 0
predlab = [0] * 2985
truelab = [0]*2985
for image, label in test_dataset:
    average_pred = average_nets(image, nets)
    pred_label = average_pred.argmax()
    true_label = label.numpy()[0]
    predlab[counter] = pred_label
    truelab[counter] = true_label
    total_correct = total_correct + (true_label == pred_label)
    counter = counter + 1
accuracy = total_correct / counter

import Confusion
Confusion.save_confusion_matrix(predlab,truelab,test_dataset.class_names,"diffensemblepredconf.png")
# Note: this method is terribly inefficient, as the tensorflow warning says. It is possible to significantly improve it.
