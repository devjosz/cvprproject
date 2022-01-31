import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np

FIRST_PART_FILENAME = "confusionfirst.png"


def predict_class_label_number(dataset):
    """Runs inference and returns predictions as class label numbers."""
    rev_label_names = {l: i for i, l in enumerate(label_names)}
    return [
        rev_label_names[o[0][0]]
        for o in model.predict_top_k(dataset, batch_size=32)
    ]


def save_confusion_matrix(predicted_labels,
                          true_labels,
                          label_names,
                          filename):
    labels = label_names
    cm = tf.math.confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(filename, format="png")


def true_labels(in_dataset):
    arr_images = []
    arr_labels = []
    for image_batch, label_batch in in_dataset:
        for label in label_batch:
            arr_labels.append(label.numpy())
    arr_labels = np.array(arr_labels)
    return arr_labels
